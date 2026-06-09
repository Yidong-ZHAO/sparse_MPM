import time
import warp as wp

class reentrant_timer:
	def __init__(self):
		self.total_time = {}     # name -> total seconds
		self.count = {}         # name -> number of entries
		self._stack = []        # support nesting if needed

	class _scope:
		def __init__(self, parent, name):
			self.parent = parent
			self.name = name
			self.t0 = None

		def __enter__(self):
			wp.synchronize()
			self.t0 = time.perf_counter()
			self.parent._stack.append(self.name)
			return self

		def __exit__(self, *args):
			wp.synchronize()
			elapsed = time.perf_counter() - self.t0
			self.parent.total_time[self.name] = (
				self.parent.total_time.get(self.name, 0.0) + elapsed
			)
			self.parent.count[self.name] = (
				self.parent.count.get(self.name, 0) + 1
			)
			self.parent._stack.pop()

	def __call__(self, name: str):
		return reentrant_timer._scope(self, name)

	def reset(self):
		self.total_time.clear()
		self.count.clear()

	def get_total_time(self, name=None):
		if name is None:
			return dict(self.total_time)
		return self.total_time.get(name, 0.0)

	def get_avg_time(self, name):
		if name not in self.total_time:
			return 0.0
		return self.total_time[name] / self.count[name]

	def get_overall_time(self):
		return sum(self.total_time.values())

	def report(self):
		lines = []
		total = self.get_overall_time()
		for name, t in self.total_time.items():
			avg = t / self.count[name]
			frac = t / total if total > 0 else 0.0
			lines.append(
				f"{name:15s}: total = {t:8.4f} s | avg = {avg:8.6f} s | {frac*100:5.1f}%"
			)
		lines.append("-" * 60)
		lines.append(f"{'TOTAL':15s}: {total:8.4f} s")
		return "\n".join(lines)