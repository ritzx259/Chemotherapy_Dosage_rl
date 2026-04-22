import math
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, Circle


class ChemoGridWorldVisualizer:
    def __init__(self, cols=6, cell_size=1.0):
        self.cols = cols
        self.cell_size = cell_size
        self.fig = None
        self.ax = None

    def _dose_color(self, dose):
        if dose < 0.2:
            return "#e8f5e9"   # low dose
        if dose < 0.5:
            return "#fff3cd"   # mild dose
        if dose < 0.8:
            return "#ffcc80"   # medium dose
        return "#ef9a9a"       # high dose

    def render_schedule(self, days, doses, tumors=None, healthy=None, title="Chemo Grid World"):
        n = len(doses)
        rows = math.ceil(n / self.cols)

        self.fig, self.ax = plt.subplots(figsize=(self.cols * 1.6, rows * 1.6))
        self.ax.set_xlim(0, self.cols)
        self.ax.set_ylim(rows, 0)
        self.ax.set_aspect("equal")
        self.ax.set_title(title)

        for i in range(n):
            r = i // self.cols
            c = i % self.cols
            dose = doses[i]
            color = self._dose_color(dose)

            rect = Rectangle((c, r), 1, 1, facecolor=color, edgecolor="black", linewidth=2)
            self.ax.add_patch(rect)

            label = f"D{int(days[i])}\nDose {dose:.2f}"

            if tumors is not None and tumors[i] < 0.05:
                label += "\nREM"

            if healthy is not None and healthy[i] < 0.3:
                label += "\nTOX"

            self.ax.text(
                c + 0.5,
                r + 0.5,
                label,
                ha="center",
                va="center",
                fontsize=8,
                weight="bold"
            )

        start_r, start_c = 0, 0
        end_r, end_c = (n - 1) // self.cols, (n - 1) % self.cols

        self.ax.add_patch(Circle((start_c + 0.18, start_r + 0.18), 0.08, color="green"))
        self.ax.text(start_c + 0.32, start_r + 0.18, "START", fontsize=8, va="center", weight="bold")

        self.ax.add_patch(Circle((end_c + 0.18, end_r + 0.82), 0.08, color="red"))
        self.ax.text(end_c + 0.35, end_r + 0.82, "END", fontsize=8, va="center", weight="bold")

        for x in range(self.cols + 1):
            self.ax.plot([x, x], [0, rows], color="black", linewidth=1)

        for y in range(rows + 1):
            self.ax.plot([0, self.cols], [y, y], color="black", linewidth=1)

        self.ax.set_xticks([])
        self.ax.set_yticks([])
        plt.tight_layout()
        plt.savefig("chemo_gridworld.png")
        plt.show()


def demo_grid_world(days, doses, tumors=None, healthy=None):
    vis = ChemoGridWorldVisualizer(cols=6)
    vis.render_schedule(days, doses, tumors=tumors, healthy=healthy)