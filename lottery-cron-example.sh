#! /bin/bash

GIT_DIR=/path/to/git/repo/for/LotteryML
VENV_DIR=/home/$USER/.local/share/virtualenvs/LotteryML-glbtLDmB/bin
CURRENT_DAY=$(date +%A)

cd $GIT_DIR
git checkout predictions

find $GIT_DIR -name "*.joblib" -delete

if [ "$CURRENT_DAY" == "Sunday" ]; then
  $VENV_DIR/python $GIT_DIR/lottery.py --gamedir NJ_Cash4Life
  $VENV_DIR/python $GIT_DIR/lottery.py --gamedir NJ_Cash5
elif [ "$CURRENT_DAY" == "Monday" ]; then
  $VENV_DIR/python $GIT_DIR/lottery.py --gamedir NJ_Cash4Life
  $VENV_DIR/python $GIT_DIR/lottery.py --gamedir NJ_Cash5
  $VENV_DIR/python $GIT_DIR/lottery.py --gamedir NJ_Pick6
  $VENV_DIR/python $GIT_DIR/lottery.py --gamedir Powerball
elif [ "$CURRENT_DAY" == "Tuesday" ]; then 
  $VENV_DIR/python $GIT_DIR/lottery.py --gamedir NJ_Cash4Life
  $VENV_DIR/python $GIT_DIR/lottery.py --gamedir NJ_Cash5
  $VENV_DIR/python $GIT_DIR/lottery.py --gamedir Megamillions
elif [ "$CURRENT_DAY" == "Wednesday" ]; then 
  $VENV_DIR/python $GIT_DIR/lottery.py --gamedir NJ_Cash4Life
  $VENV_DIR/python $GIT_DIR/lottery.py --gamedir NJ_Cash5
  $VENV_DIR/python $GIT_DIR/lottery.py --gamedir Powerball
elif [ "$CURRENT_DAY" == "Thursday" ]; then 
  $VENV_DIR/python $GIT_DIR/lottery.py --gamedir NJ_Cash4Life
  $VENV_DIR/python $GIT_DIR/lottery.py --gamedir NJ_Cash5
  $VENV_DIR/python $GIT_DIR/lottery.py --gamedir NJ_Pick6
elif [ "$CURRENT_DAY" == "Friday" ]; then 
  $VENV_DIR/python $GIT_DIR/lottery.py --gamedir NJ_Cash4Life
  $VENV_DIR/python $GIT_DIR/lottery.py --gamedir NJ_Cash5
  $VENV_DIR/python $GIT_DIR/lottery.py --gamedir Megamillions
elif [ "$CURRENT_DAY" == "Saturday" ]; then 
  $VENV_DIR/python $GIT_DIR/lottery.py --gamedir NJ_Cash4Life
  $VENV_DIR/python $GIT_DIR/lottery.py --gamedir NJ_Cash5
  $VENV_DIR/python $GIT_DIR/lottery.py --gamedir NJ_Pick6
  $VENV_DIR/python $GIT_DIR/lottery.py --gamedir Powerball
else
  echo "Error"
fi

exit
