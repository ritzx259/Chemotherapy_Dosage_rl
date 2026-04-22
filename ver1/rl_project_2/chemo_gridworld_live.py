"""import math
import time
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, Circle


class LiveChemoGridWorld:
    def __init__(self, cols=6, delay=0.6):
        self.cols = cols
        self.delay = delay
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

    def _build_grid(self, days, doses, tumors=None, healthy=None, title="Live Chemo Grid World"):
        n = len(doses)
        rows = math.ceil(n / self.cols)

        self.fig, self.ax = plt.subplots(figsize=(self.cols * 1.8, rows * 1.8))
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

            label = f"D{int(days[i])}\n{dose:.2f}"

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

        for x in range(self.cols + 1):
            self.ax.plot([x, x], [0, rows], color="black", linewidth=1)

        for y in range(rows + 1):
            self.ax.plot([0, self.cols], [y, y], color="black", linewidth=1)

        self.ax.set_xticks([])
        self.ax.set_yticks([])

        return rows

    def animate_agent(self, days, doses, tumors=None, healthy=None):
        plt.ion()
        rows = self._build_grid(days, doses, tumors=tumors, healthy=healthy)

        agent_marker = Circle((0.5, 0.5), 0.12, color="blue", zorder=10)
        self.ax.add_patch(agent_marker)

        status_text = self.ax.text(
            0.02, 1.02, "", transform=self.ax.transAxes,
            fontsize=11, weight="bold"
        )

        for i in range(len(doses)):
            r = i // self.cols
            c = i % self.cols

            agent_marker.center = (c + 0.5, r + 0.5)

            msg = f"Day {int(days[i])} | Dose {doses[i]:.2f}"
            if tumors is not None:
                msg += f" | Tumor {tumors[i]:.3f}"
            if healthy is not None:
                msg += f" | Healthy {healthy[i]:.3f}"

            status_text.set_text(msg)

            self.fig.canvas.draw()
            self.fig.canvas.flush_events()
            plt.pause(self.delay)

        plt.ioff()
        plt.show()


def live_demo_grid_world(days, doses, tumors=None, healthy=None):
    vis = LiveChemoGridWorld(cols=6, delay=0.6)
    vis.animate_agent(days, doses, tumors=tumors, healthy=healthy)
"""