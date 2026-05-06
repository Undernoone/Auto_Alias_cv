from __future__ import annotations

import argparse
from pathlib import Path


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Train the AutoAlias neural NURBS decoder.")
    parser.add_argument("json", nargs="+", type=Path)
    parser.add_argument("--out", type=Path, default=Path("checkpoints/curve_decoder.pt"))
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--hidden-dim", type=int, default=256)
    parser.add_argument("--layers", type=int, default=4)
    parser.add_argument("--heads", type=int, default=8)
    return train(parser.parse_args(argv))


def train(args: argparse.Namespace) -> int:
    import torch
    import torch.nn.functional as F
    from torch.utils.data import DataLoader

    from autoalias.learning.dataset import CurveSupervisionDataset
    from autoalias.learning.decoder import CurveTokenNURBSDecoder
    from autoalias.learning.losses import cv_distribution_loss

    device = "cuda" if torch.cuda.is_available() else "cpu"
    dataset = CurveSupervisionDataset(args.json)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, drop_last=False)
    model = CurveTokenNURBSDecoder(
        hidden_dim=args.hidden_dim,
        layers=args.layers,
        heads=args.heads,
    ).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)

    for epoch in range(1, args.epochs + 1):
        total = 0.0
        count = 0
        for batch in loader:
            points = batch["points"].to(device)
            degree_class = batch["degree_class"].to(device)
            cv_gt = batch["cv"].to(device)
            cv_mask = batch["cv_mask"].to(device).unsqueeze(-1)
            pred = model(points)
            degree_loss = F.cross_entropy(pred["degree_logits"], degree_class)
            cv_loss = (((pred["cv"] - cv_gt) * cv_mask) ** 2).sum() / cv_mask.sum().clamp_min(1.0)
            fair_loss = cv_distribution_loss(pred["cv"])
            loss = degree_loss + cv_loss + 0.01 * fair_loss
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            opt.step()
            total += float(loss.detach().cpu())
            count += 1
        print(f"epoch {epoch:03d} loss={total / max(count, 1):.6f}")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"model": model.state_dict(), "args": vars(args)}, args.out)
    print(f"saved {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

