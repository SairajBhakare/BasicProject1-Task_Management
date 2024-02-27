import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import LabelEncoder
import random

# Initialize an empty task list
tasks = pd.DataFrame(columns=['description', 'priority'])

# Load pre-existing tasks from a CSV file (if any)
try:
    tasks = pd.read_csv('tasks.csv')
except FileNotFoundError:
    pass

# Convert priority labels to numerical values
label_encoder = LabelEncoder()
tasks['priority_numeric'] = label_encoder.fit_transform(tasks['priority'])

# Function to save tasks to a CSV file
def save_tasks(tasks_df):
    tasks_df.to_csv('tasks.csv', index=False)

# Train the task priority classifier
vectorizer = CountVectorizer()
clf = MultinomialNB()
model = make_pipeline(vectorizer, clf)
model.fit(tasks['description'], tasks['priority_numeric'])

# Function to add a task to the list
def add_task(tasks_df, description, priority):
    new_task = pd.DataFrame({'description': [description], 'priority': [priority]})
    return pd.concat([tasks_df, new_task], ignore_index=True)

# Function to remove a task by description
def remove_task(tasks_df, description):
    tasks_df = tasks_df[tasks_df['description'] != description]
    return tasks_df

# Function to list all tasks
def list_tasks(tasks_df):
    if tasks_df.empty:
        print("No tasks available.")
    else:
        print(tasks_df)

# Function to recommend a task based on machine learning
def recommend_task(tasks_df):
    if not tasks_df.empty:
        high_priority_tasks = tasks_df[tasks_df['priority'] == 'High']
        
        if not high_priority_tasks.empty:
            high_priority_tasks = high_priority_tasks.reset_index(drop=True)  # Reset the index
            random_task = random.choice(high_priority_tasks['description'])
            print(f"Recommended task: {random_task} - Priority: High")
        else:
            print("No high-priority tasks available for recommendation.")
    else:
        print("No tasks available for recommendations.")

# Main menu
while True:
    print("\nTask Management App")
    print("1. Add Task")
    print("2. Remove Task")
    print("3. List Tasks")
    print("4. Recommend Task")
    print("5. Exit")

    choice = input("Select an option: ")

    if choice == "1":
        description = input("Enter task description: ")
        priority = input("Enter task priority (Low/Medium/High): ").capitalize()
        tasks = add_task(tasks, description, priority)
        save_tasks(tasks)
        print("Task added successfully.")

    elif choice == "2":
        description = input("Enter task description to remove: ")
        tasks = remove_task(tasks, description)
        save_tasks(tasks)
        print("Task removed successfully.")

    elif choice == "3":
        list_tasks(tasks)

    elif choice == "4":
        recommend_task(tasks)

    elif choice == "5":
        print("Goodbye!")
        break

    else:
        print("Invalid option. Please select a valid option.")
