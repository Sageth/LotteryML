#! /bin/bash

#! /bin/bash

GIT_DIR=/home/$USER/Documents/github/LotteryML
VENV_DIR=$(/home/$USER/.local/bin/pipenv --venv)/bin
CURRENT_DAY=$(date +%A)

cd $GIT_DIR

find $GIT_DIR -name "*.joblib" -delete

if [ "$CURRENT_DAY" == "Sunday" ]; then
  $VENV_DIR/python $GIT_DIR/lottery.py --gamedir NJ_Cash4Life --automerge
  $VENV_DIR/python $GIT_DIR/lottery.py --gamedir NJ_Cash5 --automerge
elif [ "$CURRENT_DAY" == "Monday" ]; then
  $VENV_DIR/python $GIT_DIR/lottery.py --gamedir NJ_Cash4Life --automerge
  $VENV_DIR/python $GIT_DIR/lottery.py --gamedir NJ_Cash5 --automerge
  $VENV_DIR/python $GIT_DIR/lottery.py --gamedir NJ_Pick6 --automerge
  $VENV_DIR/python $GIT_DIR/lottery.py --gamedir Powerball --automerge
elif [ "$CURRENT_DAY" == "Tuesday" ]; then 
  $VENV_DIR/python $GIT_DIR/lottery.py --gamedir NJ_Cash4Life --automerge
  $VENV_DIR/python $GIT_DIR/lottery.py --gamedir NJ_Cash5 --automerge
  $VENV_DIR/python $GIT_DIR/lottery.py --gamedir Megamillions --automerge
elif [ "$CURRENT_DAY" == "Wednesday" ]; then 
  $VENV_DIR/python $GIT_DIR/lottery.py --gamedir NJ_Cash4Life --automerge
  $VENV_DIR/python $GIT_DIR/lottery.py --gamedir NJ_Cash5 --automerge
  $VENV_DIR/python $GIT_DIR/lottery.py --gamedir Powerball --automerge
elif [ "$CURRENT_DAY" == "Thursday" ]; then 
  $VENV_DIR/python $GIT_DIR/lottery.py --gamedir NJ_Cash4Life --automerge
  $VENV_DIR/python $GIT_DIR/lottery.py --gamedir NJ_Cash5 --automerge
  $VENV_DIR/python $GIT_DIR/lottery.py --gamedir NJ_Pick6 --automerge
elif [ "$CURRENT_DAY" == "Friday" ]; then 
  $VENV_DIR/python $GIT_DIR/lottery.py --gamedir NJ_Cash4Life --automerge
  $VENV_DIR/python $GIT_DIR/lottery.py --gamedir NJ_Cash5 --automerge
  $VENV_DIR/python $GIT_DIR/lottery.py --gamedir Megamillions --automerge
elif [ "$CURRENT_DAY" == "Saturday" ]; then 
  $VENV_DIR/python $GIT_DIR/lottery.py --gamedir NJ_Cash4Life --automerge
  $VENV_DIR/python $GIT_DIR/lottery.py --gamedir NJ_Cash5 --automerge
  $VENV_DIR/python $GIT_DIR/lottery.py --gamedir NJ_Pick6 --automerge
  $VENV_DIR/python $GIT_DIR/lottery.py --gamedir Powerball --automerge
fi

exit
