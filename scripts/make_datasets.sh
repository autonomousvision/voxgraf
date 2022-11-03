SRC=$1
META=$2
DST=${3%.zip}
IFS=',' read -ra RES <<< $4

for res in ${RES[@]}
do
  OUT=${DST}_${res}.zip
  if test -f "$OUT"; then
    continue
  fi
  echo python dataset_tool.py --source $SRC  --meta $META --dest $OUT --resolution ${res}x${res} --mirror-aug True
  python dataset_tool.py --source $SRC  --meta $META --dest $OUT --resolution ${res}x${res} --mirror-aug True
done