"""
The LSTM cells we study, written by hand with nn.Linear (no nn.LSTM).

Every cell does ONE time step:
    input  : x (batch, input_size) and the previous state (h, c)
    output : the new state (h, c)

To keep the code short, one nn.Linear computes several gates at once,
then .chunk() cuts the result into one piece per gate.
For example, in the vanilla cell:
    self.x2gates(x) = [Wxi x + bi, Wxf x + bf, Wxo x + bo, Wxg x + bg]
    self.h2gates(h) = [Whi h,      Whf h,      Who h,      Whg h     ]
"""

import torch
import torch.nn as nn


class VanillaLSTMCell(nn.Module):
    """Standard LSTM, equations (1) to (6) of the guidelines."""

    def __init__(self, input_size, hidden_size):
        super().__init__()
        self.hidden_size = hidden_size  # size of c
        self.output_size = hidden_size  # size of h
        self.x2gates = nn.Linear(input_size, 4 * hidden_size)               # Wx. and b.
        self.h2gates = nn.Linear(hidden_size, 4 * hidden_size, bias=False)  # Wh.

    def forward(self, x, state):
        h, c = state
        gates = self.x2gates(x) + self.h2gates(h)
        i, f, o, g = gates.chunk(4, dim=1)
        i = torch.sigmoid(i)  # input gate   (1)
        f = torch.sigmoid(f)  # forget gate  (2)
        o = torch.sigmoid(o)  # output gate  (3)
        g = torch.tanh(g)     # candidate    (4)
        c = f * c + i * g     # new memory   (5)
        h = o * torch.tanh(c) # new output   (6)
        return h, c


class CIFGCell(nn.Module):
    """Coupled Input-Forget Gate LSTM, equations (13) to (18).
    Only difference with vanilla: there is no input gate, we use i = 1 - f.
    So we need 3 gates instead of 4 (about 25% fewer parameters)."""

    def __init__(self, input_size, hidden_size):
        super().__init__()
        self.hidden_size = hidden_size
        self.output_size = hidden_size
        self.x2gates = nn.Linear(input_size, 3 * hidden_size)
        self.h2gates = nn.Linear(hidden_size, 3 * hidden_size, bias=False)

    def forward(self, x, state):
        h, c = state
        gates = self.x2gates(x) + self.h2gates(h)
        f, o, g = gates.chunk(3, dim=1)
        f = torch.sigmoid(f)  # forget gate  (13)
        i = 1 - f             # input gate is coupled to the forget gate (14)
        o = torch.sigmoid(o)  # output gate  (15)
        g = torch.tanh(g)     # candidate    (16)
        c = f * c + i * g     # (17)
        h = o * torch.tanh(c) # (18)
        return h, c


class LSTMPCell(nn.Module):
    """LSTM with Projection, equations (19) to (25).
    Same as vanilla, but the output m (size H) is projected to a smaller
    h (size P) with an extra Linear layer Wp. h is what goes to the next
    time step and to the next layer, so the recurrent matrices are smaller."""

    def __init__(self, input_size, hidden_size, proj_size):
        super().__init__()
        self.hidden_size = hidden_size  # H, size of c and m
        self.output_size = proj_size    # P, size of h
        self.x2gates = nn.Linear(input_size, 4 * hidden_size)
        self.h2gates = nn.Linear(proj_size, 4 * hidden_size, bias=False)
        self.projection = nn.Linear(hidden_size, proj_size)  # Wp and bp

    def forward(self, x, state):
        h, c = state
        gates = self.x2gates(x) + self.h2gates(h)
        i, f, o, g = gates.chunk(4, dim=1)
        i = torch.sigmoid(i)    # (19)
        f = torch.sigmoid(f)    # (20)
        o = torch.sigmoid(o)    # (21)
        g = torch.tanh(g)       # (22)
        c = f * c + i * g       # (23)
        m = o * torch.tanh(c)   # (24)
        h = self.projection(m)  # (25)
        return h, c


class WMCCell(nn.Module):
    """LSTM with Working Memory Connections, equations (7) to (12).
    (Optional third model.) The gates also look at the memory cell c,
    through tanh(Wc c). The input and forget gates use the OLD c, the output
    gate uses the NEW c, so we compute c before the output gate."""

    def __init__(self, input_size, hidden_size):
        super().__init__()
        self.hidden_size = hidden_size
        self.output_size = hidden_size
        self.x2gates = nn.Linear(input_size, 4 * hidden_size)
        self.h2gates = nn.Linear(hidden_size, 4 * hidden_size, bias=False)
        self.c2if = nn.Linear(hidden_size, 2 * hidden_size, bias=False)  # Wci, Wcf
        self.c2o = nn.Linear(hidden_size, hidden_size, bias=False)       # Wco

    def forward(self, x, state):
        h, c_prev = state
        gates = self.x2gates(x) + self.h2gates(h)
        i, f, o, g = gates.chunk(4, dim=1)
        ci, cf = self.c2if(c_prev).chunk(2, dim=1)
        i = torch.sigmoid(i + torch.tanh(ci))  # (7)
        f = torch.sigmoid(f + torch.tanh(cf))  # (8)
        g = torch.tanh(g)                      # (10)
        c = f * c_prev + i * g                 # (11)
        o = torch.sigmoid(o + torch.tanh(self.c2o(c)))  # (9) uses the new c
        h = o * torch.tanh(c)                  # (12)
        return h, c


def make_cell(cell_type, input_size, hidden_size, proj_size=None):
    if cell_type == "vanilla":
        return VanillaLSTMCell(input_size, hidden_size)
    if cell_type == "cifg":
        return CIFGCell(input_size, hidden_size)
    if cell_type == "lstmp":
        return LSTMPCell(input_size, hidden_size, proj_size or hidden_size // 2)
    if cell_type == "wmc":
        return WMCCell(input_size, hidden_size)
    raise ValueError(f"Unknown cell type: {cell_type}")


class LSTM(nn.Module):
    """Runs a cell over a whole sequence, with several layers stacked
    (stacking cells is still a 'vanilla' model, as the guidelines say).

    x    : (batch, seq_len, input_size)
    mask : (batch, seq_len), 1 for real tokens and 0 for padding.
           On padding we keep the previous state, so padding does not
           change the final state of a sentence.
    """

    def __init__(self, cell_type, input_size, hidden_size, num_layers=1,
                 proj_size=None, dropout=0.0):
        super().__init__()
        self.cells = nn.ModuleList()
        for layer in range(num_layers):
            layer_input = input_size if layer == 0 else self.cells[-1].output_size
            self.cells.append(make_cell(cell_type, layer_input, hidden_size, proj_size))
        self.dropout = nn.Dropout(dropout)
        self.output_size = self.cells[-1].output_size

    def init_state(self, batch_size, device):
        return [(torch.zeros(batch_size, cell.output_size, device=device),
                 torch.zeros(batch_size, cell.hidden_size, device=device))
                for cell in self.cells]

    def forward(self, x, state=None, mask=None):
        batch_size, seq_len, _ = x.shape
        if state is None:
            state = self.init_state(batch_size, x.device)
        state = list(state)
        outputs = []
        for t in range(seq_len):
            inp = x[:, t]
            for layer, cell in enumerate(self.cells):
                h_old, c_old = state[layer]
                h, c = cell(inp, (h_old, c_old))
                if mask is not None:
                    m = mask[:, t].unsqueeze(1)  # (batch, 1)
                    h = m * h + (1 - m) * h_old
                    c = m * c + (1 - m) * c_old
                state[layer] = (h, c)
                inp = h if layer == len(self.cells) - 1 else self.dropout(h)
            outputs.append(inp)
        return torch.stack(outputs, dim=1), state
