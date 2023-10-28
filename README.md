# rl-study

## Homework 1

Implementing a simple Gaussian Log-Likelihood
[gaussian log-likelihood](https://spinningup.openai.com/en/latest/spinningup/exercises.html#problem-set-1-basics-of-implementation)

## Homework 2

- Create two arms (A, B) with outcomes 0 or 1 and adjustable E(P(X)). An arm that, if set to 10%, should produce a 1 10% of the time.
- Implement greedy, e-greedy, UCB, and thompson sampling. Set the hyperparameters appropriately and don't change them.
- Find the average multiplier for each algorithm by testing 5 times for the above arms, where A has a win rate of 10%, 20%, ..., 90%, and B has a win rate of 10%, 20%, ..., 90%. In total, you should test 9 X 9 X 5 X 4 times, and the final result should be 9 X 9 X 4.
- Represent the above results in a scatter plot. The x-axis is where E(P(A)) is 10%, 20%, ..., 90% and the y-axis is where E(P(B)) is 10%, 20%, ..., 90%. color is the algorithm and size is the average multiplier per algorithm.
- Submit your code to your respective GitHubs and one final image to this channel.

## Homework 3
![KakaoTalk_20231024_220614022](https://github.com/JoonHong-Kim/rl-study/assets/30318926/8c6dd8f6-2368-44cb-945f-32290e6cc78e)
Write a function that returns a value map when converging using policy evaluation and value iteration in the environment shown in the photo!
