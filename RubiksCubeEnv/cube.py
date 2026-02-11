import numpy as np
import random

from RubiksCubeEnv.utils import ActionSpace, ObservationSpace


class RubiksCube:
    """
    0 : Y (U) | 1 : R (R) | 2 : G (F) | 3 : W (D) | 4 : O (L) | 5 : B (B)
    """

    U, R, F, D, L, B = range(6)

    COLORS = {
        U: "Y",
        R: "R",
        F: "G",
        D: "W",
        L: "O",
        B: "B",
    }

    def __init__(self, n: int, _max_episode_steps = 100):
        self.n = n
        self.observation_space = ObservationSpace(0, 5, 6 * n * n, int)
        self.action_space = ActionSpace(3 * 3 * self.n) # 3 moves id, 3 variants, n layers
        self._max_episode_steps = _max_episode_steps
        self.step_count = 0
        self.state = np.zeros(self.observation_space.shape[0], dtype=np.int8)

    # --------------------------------------------------
    # Auxiliary functions
    # --------------------------------------------------

    def _face_slice(self, face_id : int):
        """
        Slice a face from the whole state vector.
        
        :param face_id: The face to be sliced.
        :type face_id: int
        """
        start = face_id * self.n * self.n
        end = (face_id + 1) * self.n * self.n
        return slice(start, end)

    def _get_face(self, face_id : int):
        """
        Return a face as a n by n matrix.

        :param face_id: The face to get.
        :type face_id: int
        """
        return self.state[self._face_slice(face_id)].reshape(self.n, self.n)

    def _set_face(self, face_id : int, face_template : np.ndarray):
        """
        Set a face of the cube using a template.

        :param face_id: The face to be sliced.
        :type face_id: int
        :param face_template: The face template.
        :type face_template: np.ndarray
        """
        self.state[self._face_slice(face_id)] = face_template.reshape(-1)

    def _rotate_face(self, face_id : int, k : int = 1):
        """
        Rotate a face 90° k-times.

        :param face_id: The face to rotate.
        :type face_id: int
        :param k: The number of rotations.
        :type k: int
        """
        face = self._get_face(face_id)
        face = np.rot90(face, -k)
        self._set_face(face_id, face)

    def _action_to_move_details(self, action : int):
        """
        Actions as integers need to be transformed to type, layer and orientation.

        :param action: An action to transform [Hyp > action is in action_space].
        :type action: int
        """
        move_id = action // (3 * self.n)
        rem = action % (3 * self.n)
        layer = rem // 3
        variant = rem % 3
        return move_id, layer, variant

    # --------------------------------------------------
    # Core moves (R, U, F generators)
    # --------------------------------------------------

    def _apply_move(self, move_id : int, layer : int, variant : int):
        """
        Take move details and apply it on the cube.

        :param move_id: One of three move type (R,U,F).
        :type move_id: int
        :param layer: The layer on which to apply the move.
        :type layer: int
        :param variant: The type of rotation to apply (Normal, Prime, Double).
        :type variant: int
        """
        if move_id == 0:
            self._move_R(layer, variant)
            return
        if move_id == 1:
            self._move_U(layer, variant)
            return
        if move_id == 2:
            self._move_F(layer, variant)
            return
        raise ValueError(f"*-* {move_id} is an invalid move type.")

    def _move_R(self, layer, variant):
        """
        Apply R move.

        :param layer: The layer on wich the move is applied (layer 0 is most right layer).
        :type layer: int
        :param variant: The type of rotation to apply (Normal, Prime, Double).
        :type variant: int
        """
        for _ in range([1, 3, 2][variant]):  # normal, prime, double
            U = self._get_face(self.U)
            F = self._get_face(self.F)
            D = self._get_face(self.D)
            B = self._get_face(self.B)

            col = self.n - 1 - layer

            tmp = U[:, col].copy()
            U[:, col] = F[:, col]
            F[:, col] = D[:, col]
            D[:, col] = B[::-1, layer]
            B[::-1, layer] = tmp

            self._set_face(self.U, U)
            self._set_face(self.F, F)
            self._set_face(self.D, D)
            self._set_face(self.B, B)

            if layer == 0:
                self._rotate_face(self.R, 1)

    def _move_U(self, layer, variant):
        """
        Apply U move.

        :param layer: The layer on wich the move is applied (layer 0 is most top layer).
        :type layer: int
        :param variant: The type of rotation to apply (Normal, Prime, Double).
        :type variant: int
        """
        for _ in range([1, 3, 2][variant]):
            F = self._get_face(self.F)
            R = self._get_face(self.R)
            B = self._get_face(self.B)
            L = self._get_face(self.L)

            tmp = F[layer, :].copy()
            F[layer, :] = R[layer, :]
            R[layer, :] = B[layer, :]
            B[layer, :] = L[layer, :]
            L[layer, :] = tmp

            self._set_face(self.F, F)
            self._set_face(self.R, R)
            self._set_face(self.B, B)
            self._set_face(self.L, L)

            if layer == 0:
                self._rotate_face(self.U, 1)

    def _move_F(self, layer, variant):
        """
        Apply F move.

        :param layer: The layer on wich the move is applied (layer 0 is most front layer).
        :type layer: int
        :param variant: The type of rotation to apply (Normal, Prime, Double).
        :type variant: int
        """
        for _ in range([1, 3, 2][variant]):
            U = self._get_face(self.U)
            R = self._get_face(self.R)
            D = self._get_face(self.D)
            L = self._get_face(self.L)

            idx = self.n - 1 - layer

            tmp = U[idx, :].copy()
            U[idx, :] = L[::-1, idx]
            L[::-1, idx] = D[layer, :]
            D[layer, :] = R[::-1, layer]
            R[::-1, layer] = tmp

            self._set_face(self.U, U)
            self._set_face(self.R, R)
            self._set_face(self.D, D)
            self._set_face(self.L, L)

            if layer == 0:
                self._rotate_face(self.F, 1)

    # --------------------------------------------------
    # Gymnasium like API
    # --------------------------------------------------

    def reward(self, state : np.ndarray = None):
        """
        Compute reward either for current state or a given state. (later can be replace with another reward function)

        :param state: A possible given state.
        :type state: np.ndarray
        """
        ret = 0
        state = self.state if not state else state
        for face in range(6):
            ret += sum(state[self._face_slice(face)] == face)
        return ret

    def step(self, action: int):
        if not self.action_space.in_bound(action):
            raise ValueError(f"*-* Action {action} if out of action space {self.action_space}")

        move_id, layer, variant = self._action_to_move_details(action)
        self._apply_move(move_id, layer, variant)
        
        reward = self.reward()
        self.step_count += 1
        is_terminated = reward == self.observation_space.shape[0] or self.step_count == self._max_episode_steps
        return np.array(self.state), reward, is_terminated, is_terminated, None

    def reset(self, scramble_length : int = 13):
        self.restart()
        actions = self.action_space.sample(scramble_length)
        for action in actions:
            self.step(action)
        return np.array(self.state), None

    def restart(self):
        for face in range(6):
            self.state[self._face_slice(face)] = face

    def show(self):
        U = self._get_face(self.U)
        L = self._get_face(self.L)
        F = self._get_face(self.F)
        R = self._get_face(self.R)
        B = self._get_face(self.B)
        D = self._get_face(self.D)

        def color(mat):
            return np.vectorize(lambda x: self.COLORS[x])(mat)

        U, L, F, R, B, D = map(color, [U, L, F, R, B, D])

        pad = " " * (self.n * 2)
        for row in U:
            print(pad + " ".join(row))
        for i in range(self.n):
            print(
                " ".join(L[i]) + "  "
                + " ".join(F[i]) + "  "
                + " ".join(R[i]) + "  "
                + " ".join(B[i])
            )
        for row in D:
            print(pad + " ".join(row))
