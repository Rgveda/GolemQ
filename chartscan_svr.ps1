CD Y:\Projects\GolemQ

# 打印 "conda activate GolemAI"
echo "conda activate GolemAI"

conda info --envs

# 激活 GolemAI 环境
C:\ProgramData\miniconda3\shell\condabin\conda-hook.ps1
conda activate 'C:\ProgramData\miniconda3'
conda activate GolemAI

$n = 1
while($n -lt 6)
{
    python -m GolemQ.cli --server ChartScan --port $args[0]
	Start-Sleep -Seconds 1
}