from RubiksCubeEnv import RubiksCube

rubik = RubiksCube(3)
rubik.restart()
print(rubik.reward())
rubik.step(0)
print(rubik.reward())
rubik.step(9)
print(rubik.reward())
# rubik.step(4)
rubik.show()
print(rubik.state)