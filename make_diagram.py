"""Draw assets/use_case_diagram.png with matplotlib."""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse, Rectangle

from modules.config import DIAGRAM_PATH
from modules.system_design import USE_CASES

fig, ax = plt.subplots(figsize=(11, 7))
ax.set_xlim(0, 11); ax.set_ylim(0, 7); ax.axis("off")
ax.add_patch(Rectangle((3, 0.3), 5.6, 6.4, fill=False, lw=1.5))
ax.text(5.8, 6.4, "Facility Reservation Risk System", ha="center", weight="bold")

ys = [5.6, 4.7, 3.8, 2.9, 2.0, 1.1]
pos = {}
for uc, y in zip(USE_CASES, ys):
    ax.add_patch(Ellipse((5.8, y), 4.2, 0.8, fill=True, fc="#e3f2fd", ec="#1565c0"))
    ax.text(5.8, y, uc, ha="center", va="center", fontsize=9)
    pos[uc] = y

actors = {"Requester": (1.2, 5.5), "Facility\nAdministrator": (1.2, 3.0), "Registrar /\nIT Admin": (10.0, 3.0)}
for name, (x, y) in actors.items():
    ax.plot(x, y + 0.5, "o", ms=14, mfc="white", mec="black")
    ax.plot([x, x], [y + 0.35, y - 0.3], "k"); ax.plot([x - .3, x + .3], [y + .1, y + .1], "k")
    ax.plot([x, x - .25], [y - .3, y - .8], "k"); ax.plot([x, x + .25], [y - .3, y - .8], "k")
    ax.text(x, y - 1.25, name, ha="center", fontsize=9)

links = {
    "Requester": ["Predict single reservation"],
    "Facility\nAdministrator": ["Explore data", "Predict single reservation",
                                "Score batch & view risk queue", "Record final decision"],
    "Registrar /\nIT Admin": ["Upload & validate dataset", "Maintain model/data", "Explore data"],
}
for a, ucs in links.items():
    ax_, ay_ = actors[a]
    for uc in ucs:
        x_end = 3.7 if ax_ < 5 else 7.9
        ax.plot([ax_ + (0.3 if ax_ < 5 else -0.3), x_end], [ay_, pos[uc]], color="#555", lw=1)

ax.text(5.5, -0.15, "Final decisions are recorded only by the Facility Administrator (human in the loop).",
        ha="center", fontsize=8, style="italic")
DIAGRAM_PATH.parent.mkdir(exist_ok=True)
fig.savefig(DIAGRAM_PATH, dpi=140, bbox_inches="tight")
print("saved", DIAGRAM_PATH)
