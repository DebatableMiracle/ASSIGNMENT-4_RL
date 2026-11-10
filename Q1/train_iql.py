"""Parameter-shared Independent Q-Learning (Double-DQN style). No global state is used anywhere in this file."""
import copy, torch, torch.nn.functional as F
from common import *

def main():
    args = get_args(); N = 5
    dev = args.device; q = AgentNet(N).to(dev); qt = copy.deepcopy(q); opt = torch.optim.Adam(q.parameters(), lr=args.lr)
    ids = torch.arange(N, device=dev).repeat(args.batch)

    def update(b, n):
        o, no, a = b["o"].reshape(-1, OBS_DIM), b["no"].reshape(-1, OBS_DIM), b["a"].reshape(-1)
        r = (b["r"] + R_TEAM * b["term"][:, None] * b["act"].float()).reshape(-1)  # own r_i (+R_team when all finish)
        act, nact = b["act"].reshape(-1).float(), b["nact"].reshape(-1).float()
        qa = q(o, ids).gather(1, a[:, None]).squeeze(1)
        with torch.no_grad():
            na = q(no, ids).argmax(1, keepdim=True)
            y = r + GAMMA * nact * qt(no, ids).gather(1, na).squeeze(1)  # finished agent => terminal for its own return
        loss = (F.smooth_l1_loss(qa, y, reduction="none") * act).sum() / act.sum().clamp(min=1)
        opt.zero_grad(); loss.backward(); torch.nn.utils.clip_grad_norm_(q.parameters(), 10); opt.step()
        if (n + 1) % args.target_every == 0: qt.load_state_dict(q.state_dict())
        return {"loss": loss.item(), "q_mean": qa.mean().item()}

    run(args, f"iql_{args.reward}_s{args.seed}", q, update)

if __name__ == "__main__": main()
