with tab2:
  import pandas as pd
  import plotly.express as px
  import numpy as np
  from sklearn.linear_model import LinearRegression
  import plotly.graph_objs as go
  # Mock data
  quiz_scores = [75, 90, 85, 78, 92, 88, 96, 81, 95, 89, 82, 91, 84, 77, 99]

  quiz_names = [
    "Introduction to AI",
    "Machine Learning Basics",
    "Deep Learning",
    "Neural Networks",
    "Computer Vision",
    "Natural Language Processing",
    "Reinforcement Learning",
    "Generative Models",
    "AI Ethics",
    "Robotics",
    "AI in Healthcare",
    "AI in Finance",
    "AI in Agriculture",
    "AI in Manufacturing",
    "AI in Transportation",
  ]

  quiz_data = pd.DataFrame({"Quiz": quiz_names, "Score": quiz_scores, "Quiz Number": list(range(1, len(quiz_scores) + 1))})

  # All quizzes chart
  st.header("Quiz Scores by Topic")

  fig = px.line_polar(quiz_data, r='Score', theta='Quiz', line_close=True, range_r=[(min(quiz_scores)-5), max(quiz_scores)])
  # Change the range_r numbers color to black
  fig.update_polars(angularaxis=dict(showline=True, linecolor="black", linewidth=2, gridcolor="white", gridwidth=1, tickfont=dict(color="black")))

  st.plotly_chart(fig, theme="streamlit", use_container_width=True)

  st.markdown("This chart visualizes the areas of study you are doing well in and the areas you should study up on.")

  # Summary chart
  average_score = quiz_data["Score"].mean()
  min_score = quiz_data["Score"].min()
  max_score = quiz_data["Score"].max()

  st.header("Summary Statistics")
  summary_data = pd.DataFrame({
    "Statistic": ["Average Score", "Lowest Score", "Highest Score"],
    "Score": [average_score, min_score, max_score],
  })
  fig = px.bar(summary_data, x="Statistic", y="Score", title="", color_discrete_sequence=["#9EE6CF"])
  fig.update_xaxes(title_text="")
  fig.update_yaxes(title_text="Score")
  st.plotly_chart(fig, theme="streamlit", use_container_width=True)

  st.markdown("This chart gives you a quick summary of your quiz scores")
  # Fit a linear regression model
  X = quiz_data["Quiz Number"].values.reshape(-1, 1)
  y = quiz_data["Score"].values.reshape(-1, 1)
  regression_model = LinearRegression()
  regression_model.fit(X, y)

  y_pred = regression_model.predict(X)

  # Quiz scores with linear regression chart
  st.header("Linear Regression of Quiz Scores")
  fig = px.scatter(quiz_data, x="Quiz Number", y="Score", text="Quiz", color_discrete_sequence=["#9EE6CF"])
  fig.add_trace(px.line(x=quiz_data["Quiz Number"], y=y_pred.reshape(-1), markers=False).data[0])
  fig.update_xaxes(title_text="Quiz Number")
  fig.update_yaxes(title_text="Score")
  st.plotly_chart(fig, theme=None, use_container_width=True)

  st.markdown("This chart shows how you are trending this semster. If the chart is trending up, you are doing continuously getting better, if the chart is trending down, you may need to study more for your quizes.")

  # Predicts next score, creates dataframe for previous score and predicted
  next_quiz_number = len(quiz_scores) + 1
  next_quiz_score = regression_model.predict(np.array([[next_quiz_number]]))[0][0]

  prev_and_projected_data = pd.DataFrame({
    "Type": ["Previous Quiz Score", "Projected Score for Next Quiz"],
    "Score": [quiz_scores[-1], next_quiz_score],
  })

  # Chart for previous and predicted
  st.header("Previous Quiz Score vs Projected Score for Next Quiz")
  fig = px.bar(prev_and_projected_data, x="Type", y="Score", title="", color_discrete_sequence=["#9EE6CF"])
  fig.update_xaxes(title_text="")
  fig.update_yaxes(title_text="Score")
  st.plotly_chart(fig, theme="streamlit", use_container_width=True)

  st.markdown("This shows you your projected score for the next quiz next to what you got on the last quiz. If the projected score is lower then what you got on the last quiz, you should impliment the same study habits for this next quiz as you did last time.")

with tab3:

  # Number of students
  num_students = 10

  # Mock data for all students
  all_students_scores = [
    [75, 90, 85, 78, 92, 88, 96, 81, 95, 89, 82, 91, 84, 99, 73],
    [68, 84, 76, 82, 89, 78, 91, 74, 92, 86, 81, 87, 79, 97, 79],
    [72, 88, 81, 77, 95, 83, 94, 80, 93, 84, 79, 90, 82, 98, 85],
    [65, 81, 74, 71, 88, 75, 87, 70, 86, 80, 75, 84, 78, 95, 69],
    [78, 92, 86, 83, 97, 89, 99, 85, 98, 90, 84, 93, 87, 100, 75],
    [71, 85, 80, 76, 90, 82, 93, 78, 91, 83, 77, 88, 81, 96, 82],
    [74, 88, 83, 80, 93, 87, 96, 82, 95, 89, 83, 92, 86, 99, 74],
    [67, 80, 75, 72, 86, 78, 89, 71, 88, 82, 76, 85, 79, 94, 83],
    [70, 84, 79, 75, 89, 81, 92, 76, 91, 85, 80, 87, 82, 97, 80],
    [73, 87, 82, 78, 92, 85, 95, 79, 94, 88, 83, 90, 85, 98, 77],
  ]

  all_students_data = []

  for i in range(num_students):
    student_data = pd.DataFrame({"Quiz": quiz_names, 
                                  "Score": all_students_scores[i], 
                                  "Quiz Number": list(range(1, len(quiz_scores) + 1)),
                                  "Student ID": [i+1]*len(quiz_scores)})
    all_students_data.append(student_data)

  all_students_data_combined = pd.concat(all_students_data, ignore_index=True)

  # Get the latest quiz scores for each student
  latest_quiz_scores = [scores[-1] for scores in all_students_scores]

  # Linear Regression of Quiz Scores for All Students
  st.header("Linear Regression of Quiz Scores for All Students")
  fig = go.Figure()

  combined_data = pd.concat(all_students_data, ignore_index=True)

  X = combined_data["Quiz Number"].values.reshape(-1, 1)
  y = combined_data["Score"].values.reshape(-1, 1)
  regression_model = LinearRegression()
  regression_model.fit(X, y)

  y_pred = regression_model.predict(X)

  for student_id, student_data in enumerate(all_students_data):
      fig.add_trace(go.Scatter(x=student_data["Quiz Number"], y=student_data["Score"], mode='markers', name=f"Student {student_id + 1}"))

  fig.add_trace(go.Scatter(x=combined_data["Quiz Number"], y=y_pred.reshape(-1), mode='lines', name="All Students Regression"))

  fig.update_xaxes(title_text="Quiz Number")
  fig.update_yaxes(title_text="Score")
  st.plotly_chart(fig, theme=None, use_container_width=True)

  st.markdown("This table shows how well the class in doing based on a trend in all of their quiz scores. If the class is trending downward you should think about giving them better resources and coming at the topics from a different approach.")

  # Create the table for performance on the last quiz
  fig = go.Figure()

  st.header("Class Quiz Scores")

  header_labels = ['Student', 'Test Name', 'Performance']
  for i, label in enumerate(header_labels):
    fig.add_shape(type='rect', xref='x', yref='y', x0=i - 0.5, x1=i + 0.5, y0=len(latest_quiz_scores), y1=len(latest_quiz_scores) + 1, fillcolor='rgba(0, 0, 0, 0.57)', line=dict(color='white'))
    fig.add_annotation(x=i, y=len(latest_quiz_scores)+ 0.25, text=label, font=dict(size=12, color='white'), showarrow=False)

  for row, score in enumerate(latest_quiz_scores):
    green_percentage = score / 100
    red_percentage = 1 - green_percentage

    # Add student information
    fig.add_shape(type='rect', xref='x', yref='y', x0=-0.5, x1=0.5, y0=row, y1=row + 1, fillcolor='white', line=dict(color='white'))
    fig.add_annotation(x=0, y=row + 0.5, text=f"Student {row + 1}", font=dict(size=11, color='black'), showarrow=False)

    # Add test name
    fig.add_shape(type='rect', xref='x', yref='y', x0=0.5, x1=1.5, y0=row, y1=row + 1, fillcolor='white', line=dict(color='white'))
    fig.add_annotation(x=1, y=row + 0.5, text=quiz_names[-1], font=dict(size=11, color='black'), showarrow=False)

    # Add performance
    fig.add_shape(type='rect', xref='x', yref='y', x0=1.5, x1=1.5 + green_percentage, y0=row, y1=row + 1, fillcolor='rgba(118, 255, 162, 0.77)', line=dict(color='white'))
    fig.add_shape(type='rect', xref='x', yref='y', x0=1.5 + green_percentage, x1=2.5, y0=row, y1=row + 1, fillcolor='rgba(255, 91, 52, 0.57)', line=dict(color='white'))
    fig.add_annotation(x=2, y=row + 0.5, text=f"{score}%", font=dict(size=11, color='black'), showarrow=False)

  fig.update_xaxes(showgrid=False, zeroline=False, visible=False, range=[-0.5, 2.5])
  fig.update_yaxes(showgrid=False, zeroline=False, visible=False, range=[-0.5, len(latest_quiz_scores) + 0.5], autorange='reversed')
  fig.update_layout(title='Latest Quiz Performance', width=800, height=40 * (len(latest_quiz_scores) + 1), margin=dict(t=50, b=0, l=0, r=0))
  st.plotly_chart(fig, theme="streamlit", use_container_width=True)
  st.markdown("This table shows how each student performed on the last quiz. This will help you understand which students are underperforming so you can reach out to them.")
