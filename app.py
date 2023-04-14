with tab2:
  import pandas as pd
  import plotly.express as px
  import numpy as np
  from sklearn.linear_model import LinearRegression

  # Mock data for quiz scores
  quiz_scores = [75, 90, 85, 78, 92, 88, 96, 81, 95, 89, 82, 91, 84, 77, 99]

  # Quiz names related to AI
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

  # Create a DataFrame with quiz scores
  quiz_data = pd.DataFrame({"Quiz": quiz_names, "Score": quiz_scores, "Quiz Number": list(range(1, len(quiz_scores) + 1))})

  # Calculate summary statistics
  average_score = quiz_data["Score"].mean()
  min_score = quiz_data["Score"].min()
  max_score = quiz_data["Score"].max()

  # Display summary statistics as a bar chart
  st.header("Summary Statistics")
  summary_data = pd.DataFrame({
    "Statistic": ["Average Score", "Lowest Score", "Highest Score"],
    "Score": [average_score, min_score, max_score],
  })
  fig = px.bar(summary_data, x="Statistic", y="Score", title="", color_discrete_sequence=["#9EE6CF"])
  fig.update_xaxes(title_text="")
  fig.update_yaxes(title_text="Score")
  st.plotly_chart(fig, theme="streamlit", use_container_width=True)

  # Visualize quiz scores using a bar chart
  st.header("Quiz Scores by AI Topic")
  fig = px.bar(quiz_data, x="Quiz", y="Score", title="", color_discrete_sequence=["#9EE6CF"])
  fig.update_xaxes(title_text="Quiz")
  fig.update_yaxes(title_text="Score")
  st.plotly_chart(fig, theme="streamlit", use_container_width=True)

  # Fit a linear regression model
  X = quiz_data["Quiz Number"].values.reshape(-1, 1)
  y = quiz_data["Score"].values.reshape(-1, 1)
  regression_model = LinearRegression()
  regression_model.fit(X, y)

  # Generate predictions for linear regression line
  y_pred = regression_model.predict(X)

  # Visualize quiz scores with a linear regression line
  st.header("Linear Regression of Quiz Scores")
  fig = px.scatter(quiz_data, x="Quiz Number", y="Score", text="Quiz", color_discrete_sequence=["#9EE6CF"])
  fig.add_trace(px.line(x=quiz_data["Quiz Number"], y=y_pred.reshape(-1), markers=False).data[0])
  fig.update_xaxes(title_text="Quiz Number")
  fig.update_yaxes(title_text="Score")
  st.plotly_chart(fig, theme="streamlit", use_container_width=True)

  # Project the score for the next quiz
  next_quiz_number = len(quiz_scores) + 1
  next_quiz_score = regression_model.predict(np.array([[next_quiz_number]]))[0][0]

  # Create a DataFrame for previous and projected scores
  prev_and_projected_data = pd.DataFrame({
    "Type": ["Previous Quiz Score", "Projected Score for Next Quiz"],
    "Score": [quiz_scores[-1], next_quiz_score],
  })

  # Display the previous quiz score and projected score for the next quiz as a bar chart
  st.header("Previous Quiz Score vs Projected Score for Next Quiz")
  fig = px.bar(prev_and_projected_data, x="Type", y="Score", title="", color_discrete_sequence=["#9EE6CF"])
  fig.update_xaxes(title_text="")
  fig.update_yaxes(title_text="Score")
  st.plotly_chart(fig, theme="streamlit", use_container_width=True)
