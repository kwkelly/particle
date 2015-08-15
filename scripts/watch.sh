#!/bin/bash

while true
do
  buffer=$(
    clear
		tac $1 | less -r
  )
  echo "$buffer"
  sleep 2
done
