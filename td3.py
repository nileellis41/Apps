import pandas as pd
import streamlit as st
import json
import os

# Initialize the to-do list DataFrame
columns = ['Check', 'Index', 'Project', 'Task', 'Priority Level']
todo_list = pd.DataFrame(columns=columns)

# Load existing tasks from session state
if 'todo_list' not in st.session_state:
    st.session_state.todo_list = todo_list

# Load tasks from a JSON file
def load_tasks(file_name):
    if os.path.exists(file_name):
        with open(file_name, 'r') as file:
            data = json.load(file)
            st.session_state.todo_list = pd.DataFrame(data)

# Save tasks to a JSON file
def save_tasks(file_name):
    st.session_state.todo_list.to_json(file_name, orient='records')

# Save tasks to a CSV file
def save_tasks_csv(file_name):
    st.session_state.todo_list.to_csv(file_name, index=False)

# Function to add a task
def add_task(project, task, priority):
    new_index = len(st.session_state.todo_list) + 1  # Create a new index
    new_task = pd.DataFrame({'Check': ['[ ]'], 'Index': [new_index], 'Project': [project], 'Task': [task], 'Priority Level': [priority]})
    st.session_state.todo_list = pd.concat([st.session_state.todo_list, new_task], ignore_index=True)

# Function to mark a task as complete
def mark_task_complete(index):
    if 1 <= index <= len(st.session_state.todo_list):
        st.session_state.todo_list.at[index - 1, 'Check'] = '[x]'  # Mark task as complete

# Function to suggest an optimal task order based on priority
def prioritize_tasks():
    priorities = {'High': 1, 'Medium': 2, 'Low': 3}
    st.session_state.todo_list['Priority Order'] = st.session_state.todo_list['Priority Level'].map(priorities)
    sorted_list = st.session_state.todo_list.sort_values(by=['Priority Order'], ascending=True)
    return sorted_list

# Streamlit App
st.title("To-Do List Application")

# Load tasks if any
load_tasks('todo_list.json')

# Input fields to add a new task
with st.form("Add Task Form"):
    project = st.text_input("Project Name")
    task = st.text_input("Task Description")
    priority = st.selectbox("Priority Level", ["High", "Medium", "Low"])
    submit_button = st.form_submit_button("Add Task")

    if submit_button:
        add_task(project, task, priority)
        st.success("Task added!")

# Display the prioritized current to-do list
st.subheader("Current To-Do List")
if not st.session_state.todo_list.empty:
    sorted_tasks = prioritize_tasks()
    st.table(sorted_tasks[['Check', 'Index', 'Project', 'Task', 'Priority Level']])

    # Option to mark tasks as complete
    index_to_complete = st.number_input("Enter Task Index to Complete", min_value=1, max_value=len(st.session_state.todo_list), step=1)
    if st.button("Complete Task"):
        mark_task_complete(index_to_complete)
        st.success("Task marked as complete!")
else:
    st.write("No tasks available. Please add a task.")

# Save tasks to a JSON file
if st.button("Save Tasks to JSON"):
    save_tasks('todo_list.json')
    st.success("Tasks saved successfully!")

# Save tasks to a CSV file
if st.button("Save Tasks to CSV"):
    save_tasks_csv('todo_list.csv')
    st.success("Tasks saved as CSV successfully!")

# Display the updated to-do list
st.subheader("Updated To-Do List")
st.table(st.session_state.todo_list[['Check', 'Index', 'Project', 'Task', 'Priority Level']])