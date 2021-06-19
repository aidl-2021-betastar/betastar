from multiprocessing import Pipe, Process
from multiprocessing.connection import Connection
from typing import Optional

import torch as T
from betastar.envs.env import PySC2Env, spawn_env

START, STEP, RESET, STOP, DONE = range(5)


class ProcEnv(object):
    environment: str
    game_speed: int
    _env: Optional[PySC2Env]
    conn: Optional[Connection]
    w_conn: Optional[Connection]
    proc: Optional[Process]

    def __init__(self, environment: str, game_speed: int) -> None:
        super(ProcEnv).__init__()
        self.environment = environment
        self.game_speed = game_speed
        self._env = self.conn = self.w_conn = self.proc = None

    def start(self):
        self.conn, self.w_conn = Pipe()
        self.proc = Process(target=self._run)
        self.proc.start()
        self.conn.send((START, None))

    def step(self, act):
        self.conn.send((STEP, act))

    def reset(self):
        self.conn.send((RESET, None))

    def stop(self):
        self.conn.send((STOP, None))

    def wait(self):
        return self.conn.recv()

    def _run(self):
        while True:
            msg, data = self.w_conn.recv()
            if msg == START:
                self._env = spawn_env(self.environment, self.game_speed, monitor=False)
                self.w_conn.send(DONE)
            elif msg == STEP:
                observation, reward, done, _info = self._env.step(data)
                self.w_conn.send((observation, reward, done, self._env.action_mask))
            elif msg == RESET:
                observation = self._env.reset()
                self.w_conn.send((observation, -1, False, self._env.action_mask))
            elif msg == STOP:
                self._env.close()
                self.w_conn.close()
                break


class MultiProcEnv(object):
    def __init__(self, environment: str, game_speed: int, count: int):
        super(MultiProcEnv).__init__()
        self.envs = [ProcEnv(environment, game_speed) for env in range(count)]

    def start(self):
        for env in self.envs:
            env.start()
        self.wait()

    def step(self, actions):
        for idx, env in enumerate(self.envs):
            env.step(actions[idx])
        return self._observe()

    def reset(self, only=None):
        for idx, e in enumerate(self.envs):
            if only is None or idx in only:
                e.reset()
        return self._observe(only=only)

    def _observe(self, only=None):
        obs, reward, done, action_mask = zip(*self.wait(only=only))
        return T.stack([o[0] for o in obs]), T.stack([o[1] for o in obs]), T.stack([o[2] for o in obs]), T.tensor(reward), T.tensor(done), T.stack([T.from_numpy(a) for a in action_mask])

    def stop(self):
        for e in self.envs:
            e.stop()
        for e in self.envs:
            e.proc.join()

    def wait(self, only=None):
        return [e.wait() for idx, e in enumerate(self.envs) if only is None or idx in only]
