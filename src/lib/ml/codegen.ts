import { TrainConfig } from "./engine";

export function pytorchCode(names: string[], target: string, degree: number, cfg: TrainConfig): string {
  const d = names.length;
  const build =
    degree === 1
      ? "# columns of X, in order: " + names.join(", ")
      : "# columns of X (original predictors plus powers and products), in order:\n# " + names.join(", ");
  return `import torch
import torch.nn as nn

${build}
# X: standardized, shape (N, ${d});  y: "${target}" standardized, shape (N, 1)

model = nn.Linear(${d}, 1)          # y_hat = b + w1*x1 + ... + w${d}*x${d}
loss_fn = nn.MSELoss()              # L = mean((y - y_hat)^2)
optimizer = torch.optim.Adam(model.parameters(), lr=${cfg.lr})

for epoch in range(${cfg.epochs}):
    optimizer.zero_grad()
    loss = loss_fn(model(X), y)
    loss.backward()                 # computes the partial derivatives dL/dw
    optimizer.step()                # moves each weight downhill
`;
}
