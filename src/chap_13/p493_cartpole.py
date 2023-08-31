import gym

env = gym.make("CartPole-v1")		# (1)
observation = env.reset()		# (2)

for _ in range(1000):			# (3)	
  env.render()				# (4)
  action = env.action_space.sample() 		# (5)
  observation, reward, done, info = env.step(action)	# (6)

  if done:
    observation = env.reset()		# (7)
env.close()


