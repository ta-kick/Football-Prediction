# Football Prediction
This is DeepLearning model to predict football game on keras in python.  
<br>
Python kerasを用いたサッカーの試合勝敗予想

## Description
- Using Keras in python.
- CSV file has information including Opponent's team name, Win or Draw or Lose, Home game or not, self lanking as of the game, opponent's  lanking for three games as of the game.
- When it trains, it use opponent's team name, Win or Lose, Home game or not, self lanking as of the game and opponent's lanking as of the game information this time.
- It is Information for the past twenty years.
- This predict Espanyol's game in La Liga(Spanish League) 2018-2019 season. (*exclude tie games.)
- DeepLearning Models are FFNN, LSTM and GRU.  

## 概要(Description)
- PythonのKerasを使用。
- 学習時に使うCSVファイルは、相手のチーム名、勝or負or引き分け、ホームかアウェイか、その試合までの自チーム順位、相手の過去3試合分の順位の情報を持つ。
- 今回、学習時に使ったのは、相手のチーム名、勝or負or引き分け、ホームかアウェイか、その試合までの自チーム順位、その試合までの相手の順位の情報。
- データは過去20年分。
- スペインリーグLa Liga　2018-2019シーズンのエスパニョールの試合結果を予測した。
- DeepLearningのmodelはFFNN、LSTM、GRU。
<br>
<br>
<br>
  
  
  
![result](https://user-images.githubusercontent.com/49090703/57839151-1985a480-7801-11e9-967c-d642c4bf81a5.PNG)

## System Requirement
- python3.7.1
- keras 2.2.4
- numpy 1.15.4
- scikit-learn 0.20.1
- pandas 0.23.4
- jupyter notebook 5.7.4
- Anaconda Navigator 1.9.6
- windows10
