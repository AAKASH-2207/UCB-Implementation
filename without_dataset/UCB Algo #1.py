'''
this code is created by use of blackbox AI and got to working condition by Aakash Sharma
Date: Saturday, 11th May 2024
'''
import numpy as np
import matplotlib.pyplot as plt

def UCB_algorithm(
    # Previous rewards
    rewards = [0]

    # Number of times each campaign was chosen
    selections = [0, 0, 0]

    # Total number of tests
    total_tests = 0

    # Number of tests to perform
    num_tests = 1000

    # Number of campaigns to choose from
    num_campaigns = len(rewards)

    # Exploration rate
    alpha = 0.5

    # Function to calculate upper confidence bounds
    def ucb(mean, num_tests, total_tests):
        return mean + alpha * np.sqrt(np.log(total_tests + 1) / (num_tests + 1))

    # Main loop of algorithm
    for _ in range(num_tests):
        # Calculate upper confidence bounds for each campaign
        ucb_values = np.array([ucb(rewards[i], selections[i], total_tests) for i in range(num_campaigns)])
    
        # Choose a campaign based on the UCB algorithm
        chosen_campaign = np.argmax(ucb_values)
    
        # Increment the total number of tests
        total_tests += 1
    
    # Update the number of times the chosen campaign was selected
        selections[chosen_campaign] += 1
    
    # Calculate the reward for the chosen campaign
        reward = np.random.beta(selections[chosen_campaign], 1)
    
    # Update the reward for the chosen campaign
        rewards[chosen_campaign] += reward
    
    # Print the results
        print("Test {}: Chosen campaign: {}, Reward: {:.2f}, Upper Confidence Bounds: {}".format(_, chosen_campaign, reward, ucb_values))
