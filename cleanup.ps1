# 脚本用于清理不需要的工具脚本

# 清理列表中包含所有要删除的文件
$filesToRemove = @(
    "create_structure.bat",
    "create_structure.ps1",
    "move_files.bat",
    "move_files.ps1"
)

# 遍历并删除文件
foreach ($file in $filesToRemove) {
    if (Test-Path -Path $file) {
        Remove-Item -Path $file -Force
        Write-Host "已删除: $file" -ForegroundColor Green
    }
    else {
        Write-Host "文件不存在: $file" -ForegroundColor Yellow
    }
}

Write-Host "清理完成!" -ForegroundColor Cyan
