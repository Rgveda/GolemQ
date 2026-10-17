CD Y:\Projects\GolemQ

# 打印 "conda activate GolemQ"
echo "conda activate GolemQ"

conda info --envs

# 激活 GolemQ 环境
C:\ProgramData\miniconda3\shell\condabin\conda-hook.ps1
conda activate 'C:\ProgramData\miniconda3'
conda activate GolemQ

$n = 1
while($n -lt 6)
{
    # 执行 Python 命令，并传递参数 $1
    python -m GolemQ.cli --scan ChartScan --eval $args[0] --port $args[1]
	Start-Sleep -Seconds 1
}