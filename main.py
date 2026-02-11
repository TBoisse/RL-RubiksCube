from RubiksCubeEnv import RubiksCube

rubik = RubiksCube(3)
rubik.restart()
obs, reward, terminated, truncated, info = rubik.step(0)
rubik.show()