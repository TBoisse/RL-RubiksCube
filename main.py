from RubiksCubeEnv import RubiksCube

rubik = RubiksCube(3)
rubik.restart()
obs, reward, terminated, truncated, info = rubik.step(0)
rubik.show()

i = 0
while terminated == False:
    print(i)
    a = rubik.action_space.sample()[0]
    _, _, terminated, _, _ = rubik.step(a)
    i += 1
