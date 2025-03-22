i = $1
while true
do
    sh gan.sh "$i"
    ((i++))
    echo "$1 次循环完成"
done
