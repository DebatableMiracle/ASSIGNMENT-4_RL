"""QMIX. Actors/Q-networks see (o_i, id_i) only. The global state enters ONLY the hypernetwork Mixer, used only in `update`
(training). The acting/eval path (common.run / greedy) calls AgentNet.forward(obs, idx), which has no state argument."""
import copy, torch, torch.nn as nn, torch.nn.functional as F
from common import *

class Mixer(nn.Module):
    def __init__(self, n, sdim, emb=32, hyp=64):
        super().__init__()
        self.n, self.emb = n, emb
        self.w1 = nn.Sequential(nn.Linear(sdim, hyp), nn.ReLU(), nn.Linear(hyp, n * emb)); self.b1 = nn.Linear(sdim, emb)
        self.w2 = nn.Sequential(nn.Linear(sdim, hyp), nn.ReLU(), nn.Linear(hyp, emb))
        self.b2 = nn.Sequential(nn.Linear(sdim, emb), nn.ReLU(), nn.Linear(emb, 1))

    def forward(self, qs, s):  # qs (B,n); abs() weights => dQtot/dQi >= 0
        w1, b1 = self.w1(s).abs().view(-1, self.n, self.emb), self.b1(s).view(-1, 1, self.emb)
        h = F.elu(torch.bmm(qs.unsqueeze(1), w1) + b1)
        return (torch.bmm(h, self.w2(s).abs().view(-1, self.emb, 1)) + self.b2(s).view(-1, 1, 1)).view(-1)

def main():
    args = get_args(); N = 5; dev = args.device; sdim = FastEnv().state_dim
    q, mix = AgentNet(N).to(dev), Mixer(N, sdim).to(dev); qt, mixt = copy.deepcopy(q), copy.deepcopy(mix)
    opt = torch.optim.Adam(list(q.parameters()) + list(mix.parameters()), lr=args.lr)
    ids = torch.arange(N, device=dev).repeat(args.batch)

    def update(b, n):
        B = args.batch; act, nact = b["act"].float(), b["nact"].float()
        qa = q(b["o"].reshape(-1, OBS_DIM), ids).gather(1, b["a"].reshape(-1, 1)).view(B, N) * act  # finished agents -> 0
        qtot = mix(qa, b["s"])
        with torch.no_grad():
            no = b["no"].reshape(-1, OBS_DIM); na = q(no, ids).argmax(1, keepdim=True)  # double-Q action selection
            nq = qt(no, ids).gather(1, na).view(B, N) * nact
            y = b["tr"] + GAMMA * (1 - b["term"]) * mixt(nq, b["ns"])  # team reward r^t (incl. R_team)
        loss = F.smooth_l1_loss(qtot, y)
        opt.zero_grad(); loss.backward(); torch.nn.utils.clip_grad_norm_(list(q.parameters()) + list(mix.parameters()), 10); opt.step()
        if (n + 1) % args.target_every == 0: qt.load_state_dict(q.state_dict()); mixt.load_state_dict(mix.state_dict())
        return {"loss": loss.item(), "qtot_mean": qtot.mean().item()}

    run(args, f"qmix_{args.reward}_s{args.seed}", q, update, save_extra=lambda: {"mixer": mix.state_dict()})

if __name__ == "__main__": main()
