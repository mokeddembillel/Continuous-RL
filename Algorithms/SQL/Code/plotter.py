import numpy as np
import matplotlib.pyplot as plt
import torch as T

class QFPolicyPlotter:
    def __init__(self, qf, policy, obs_lst, default_action, n_samples):
        self._qf = qf
        self._policy = policy
        self._obs_lst = obs_lst
        self._default_action = default_action
        self._n_samples = n_samples

        self._var_inds = np.where(np.isnan(default_action))[0]
        assert len(self._var_inds) == 2

        n_plots = len(obs_lst)

        x_size = 5 * n_plots
        y_size = 5

        self.fig = plt.figure(figsize=(x_size, y_size))
        self._ax_lst = []
        for i in range(n_plots):
            ax = self.fig.add_subplot(100 + n_plots * 10 + i + 1)
            ax.set_xlim((-1, 1))
            ax.set_ylim((-1, 1))
            ax.grid(True)
            self._ax_lst.append(ax)

        self._line_objects = list()

    def draw(self):
        # noinspection PyArgumentList
        [h.remove() for h in self._line_objects]
        self._line_objects = list()

        self._plot_level_curves()

        self._plot_action_samples()

        plt.draw()
        plt.pause(0.001)

    def _plot_level_curves(self):
        # Create mesh grid.
        xs = np.linspace(-1, 1, 50)
        ys = np.linspace(-1, 1, 50)
        xgrid, ygrid = np.meshgrid(xs, ys)
        N = len(xs)*len(ys)

        # Copy default values along the first axis and replace nans with
        # the mesh grid points.
        actions = np.tile(self._default_action, (N, 1))
        actions[:, self._var_inds[0]] = xgrid.ravel()
        actions[:, self._var_inds[1]] = ygrid.ravel()
        actions = T.from_numpy(actions.astype(np.float32))
        for ax, obs in zip(self._ax_lst, self._obs_lst):
            obs = T.FloatTensor(obs).repeat([actions.shape[0],1])
            with T.no_grad():
                qs = self._qf(obs.to(self._qf.device), actions.to(self._qf.device)).cpu().detach().numpy()

            qs = qs.reshape(xgrid.shape)

            cs = ax.contour(xgrid, ygrid, qs, 20)
            self._line_objects += cs.collections
            self._line_objects += ax.clabel(
                cs, inline=1, fontsize=10, fmt='%.2f')

    def _plot_action_samples(self):
        for ax, obs in zip(self._ax_lst, self._obs_lst):
            with T.no_grad():
                actions = self._policy(T.FloatTensor(obs).repeat([self._n_samples,1]).to(self._policy.device)).cpu().detach().numpy()
            x, y = actions[:, 0], actions[:, 1]
            ax.title.set_text(str(obs))
            self._line_objects += ax.plot(x, y, 'b*')