import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import matplotlib.animation as animation
import numpy as np

from DecoderRun import Decoder, GeneralDecoder, ThreeByThreeDecoder
from RubiksCube.basics import RotationType, MoveType, opposite_rotation

def get_decoder(size : int) -> Decoder:
    if size == 3:
        return ThreeByThreeDecoder()
    return GeneralDecoder()

def get_u_order(rotation_type : RotationType):
    if rotation_type == RotationType.PRIME:
        return {1 : 4, 2 : 1, 3 : 2, 4 : 3}
    if rotation_type == RotationType.DOUBLE:
        return {1 : 3, 2 : 4, 3 : 1, 4 : 2}
    return {1 : 2, 2 : 3, 3 : 4, 4 : 1}

class RubiksCube():
    def __init__(self, size, decoder : Decoder = None):
        # self.size * 0 = white, 1 = green, 2 = orange, 3 = blue, 4 = red, 5 = yellow
        # white bottom and green front
        self.size = size
        self.decoder = get_decoder(size) if decoder is None else decoder
        
        # generate color map
        self.cmap = mcolors.ListedColormap(["black", "white", "green", "orange", "blue", "red", "yellow"])
        bounds = [-1.5, -0.5, 0.5, 1.5, 2.5, 3.5, 4.5, 5.5]
        self.norm = mcolors.BoundaryNorm(bounds, self.cmap.N)
        # generate faces
        self.faces = np.ones((self.size * 3, self.size * 4)) * -1
        self.faces_bounds = {
            0: [self.size * 2, self.size * 3, self.size * 1, self.size * 2],
            1: [self.size * 1, self.size * 2, self.size * 1, self.size * 2],
            2: [self.size * 1, self.size * 2, self.size * 2, self.size * 3],
            3: [self.size * 1, self.size * 2, self.size * 3, self.size * 4],
            4: [self.size * 1, self.size * 2, self.size * 0, self.size * 1],
            5: [self.size * 0, self.size * 1, self.size * 1, self.size * 2],
        }
        self.faces[self.faces_bounds[5][0]:self.faces_bounds[5][1],self.faces_bounds[5][2]:self.faces_bounds[5][3]] = 5 # set yellow
        self.faces[self.faces_bounds[4][0]:self.faces_bounds[4][1],self.faces_bounds[4][2]:self.faces_bounds[4][3]] = 4 # set red
        self.faces[self.faces_bounds[1][0]:self.faces_bounds[1][1],self.faces_bounds[1][2]:self.faces_bounds[1][3]] = 1 # set green
        self.faces[self.faces_bounds[2][0]:self.faces_bounds[2][1],self.faces_bounds[2][2]:self.faces_bounds[2][3]] = 2 # set orange
        self.faces[self.faces_bounds[3][0]:self.faces_bounds[3][1],self.faces_bounds[3][2]:self.faces_bounds[3][3]] = 3 # set blue
        self.faces[self.faces_bounds[0][0]:self.faces_bounds[0][1],self.faces_bounds[0][2]:self.faces_bounds[0][3]] = 0 # set white

    def rotate(self, face_index : int, rotation_number : int):
        assert 0 <= face_index < 6, "*-* Rotate error : Index out of range."
        self.faces[self.faces_bounds[face_index][0]:self.faces_bounds[face_index][1],self.faces_bounds[face_index][2]:self.faces_bounds[face_index][3]] = \
            np.rot90(self.faces[self.faces_bounds[face_index][0]:self.faces_bounds[face_index][1],self.faces_bounds[face_index][2]:self.faces_bounds[face_index][3]], rotation_number)

    def __u_move(self, move_index : int, rotation_type : RotationType):
        assert 0 <= move_index < self.size, "*-* U move error : Index out of range."
        if move_index == 0:
            self.rotate(5, rotation_type.value)
        if move_index == self.size - 1:
            self.rotate(0, opposite_rotation(rotation_type).value)
        row = self.faces[self.size + move_index]
        if rotation_type == RotationType.NORMAL:
            row = np.concat((row[self.size:], row[:self.size]))
        elif rotation_type == RotationType.PRIME:
            row = np.concat((row[self.size * 3:], row[:3*self.size]))
        elif rotation_type == RotationType.DOUBLE:
            row = np.concat((row[self.size * 2:], row[:2*self.size]))
        self.faces[self.size + move_index] = row
        
    def __r_move(self, move_index : int, rotation_type : RotationType):
        assert 0 <= move_index < self.size, "*-* R move error : Index out of range."
        if move_index == 0:
            self.rotate(4, opposite_rotation(rotation_type).value)
        if move_index == self.size - 1:
            self.rotate(2, rotation_type.value)
        row = np.concat((self.faces[:, self.size + move_index], self.faces[self.size:self.size * 2, self.size * 4 - 1 - move_index][::-1]))
        if rotation_type == RotationType.NORMAL:
            row = np.concat((row[self.size:], row[:self.size]))
        elif rotation_type == RotationType.PRIME:
            row = np.concat((row[self.size * 3:], row[:3*self.size]))
        elif rotation_type == RotationType.DOUBLE:
            row = np.concat((row[self.size * 2:], row[:2*self.size]))
        self.faces[:self.size * 3, self.size + move_index] = row[:self.size * 3]
        self.faces[self.size:self.size * 2, self.size * 4 - 1 - move_index] = row[self.size * 3:][::-1]
        
    def __f_move(self, move_index : int, rotation_type : RotationType):
        assert 0 <= move_index < self.size, "*-* F move error : Index out of range."
        if move_index == 0:
            self.rotate(1, rotation_type.value)
        if move_index == self.size - 1:
            self.rotate(3, opposite_rotation(rotation_type).value)
        row = np.concat((
            self.faces[self.size:self.size*2, self.size - 1 - move_index][::-1], # red
            self.faces[self.size - 1 - move_index, self.size:self.size*2], # yellow
            self.faces[self.size:self.size*2, self.size * 2 + move_index], # orange
            self.faces[self.size * 2 + move_index, self.size:self.size*2][::-1] # white
        ))
        if rotation_type == RotationType.NORMAL:
            row = np.concat((row[self.size * 3:], row[:3*self.size]))
        elif rotation_type == RotationType.PRIME:
            row = np.concat((row[self.size:], row[:self.size]))
        elif rotation_type == RotationType.DOUBLE:
            row = np.concat((row[self.size * 2:], row[:2*self.size]))
        self.faces[self.size:self.size*2, self.size - 1 - move_index] = row[:self.size][::-1] # red
        self.faces[self.size - 1 - move_index, self.size:self.size*2] = row[self.size:self.size*2] # yellow
        self.faces[self.size:self.size*2, self.size * 2 + move_index] = row[self.size*2:self.size*3] # orange
        self.faces[self.size * 2 + move_index, self.size:self.size*2] = row[self.size*3:][::-1] # white

    def move(self, move_type : MoveType, move_index : int, rotation_type : RotationType):
        if move_type == MoveType.U:
            self.__u_move(move_index, rotation_type)
            return
        if move_type == MoveType.F:
            self.__f_move(move_index, rotation_type)
            return
        self.__r_move(move_index, rotation_type)
        return
    
    def run(self, sequence : str):
        to_run = self.decoder.split(sequence)
        for elm in to_run:
            move_type, move_index, rotation_type = self.decoder.decode(elm)
            self.move(move_type, move_index, rotation_type)

    def animate(self, sequence : str, fps = 1, pause_frames = 2):
        # prepare fig
        fig = plt.figure(figsize=(6,6))
        # prepare sequence to run
        to_run = self.decoder.split(sequence)
        # prepare cube
        img = plt.imshow(self.faces, cmap=self.cmap, norm=self.norm)
        for i in range(self.size * 3):
            plt.plot([-0.5, self.size * 4 - 0.5], [i - 0.5, i - 0.5], 'k', linewidth=1)
        for i in range(self.size * 4):
            plt.plot([i - 0.5, i - 0.5], [-0.5, self.size * 3 - 0.5], 'k', linewidth=1)
        plt.xticks([])
        plt.yticks([])
        # init animation
        def init():
            return img
        # update animation
        def update(frames):
            if frames == -1:
                return img
            move_type, move_index, rotation_type = self.decoder.decode(to_run[frames])
            self.move(move_type, move_index, rotation_type)
            img.set_data(self.faces)
            return img
        # save animation
        ani = animation.FuncAnimation(fig, update, frames=list(range(-1, len(to_run))) + [-1]*pause_frames,
                            init_func=init, blit=False, interval=1000)
        ani.save("animation.gif", writer="pillow", fps=fps)

    @staticmethod
    def from_cube(faces):
        ret = RubiksCube(len(faces) // 3)
        for i in range(len(faces)):
            for j in range(len(faces[0])):
                ret.faces[i][j] = faces[i][j]
        return ret
        

    def show(self):
        plt.figure(figsize=(6,6))
        # show cube
        plt.imshow(self.faces, cmap=self.cmap, norm=self.norm)
        # segment cube
        for i in range(self.size * 3):
            plt.plot([-0.5, self.size * 4 - 0.5], [i - 0.5, i - 0.5], 'k', linewidth=1)
        for i in range(self.size * 4):
            plt.plot([i - 0.5, i - 0.5], [-0.5, self.size * 3 - 0.5], 'k', linewidth=1)
        plt.xticks([])
        plt.yticks([])
        plt.show()