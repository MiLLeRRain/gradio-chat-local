# 脚本用于整理文件到scripts目录

# 创建scripts目录
$scriptsDir = "scripts"
if (-not (Test-Path -Path $scriptsDir)) {
    New-Item -ItemType Directory -Path $scriptsDir | Out-Null
    Write-Host "创建目录: $scriptsDir" -ForegroundColor Green
}

# 要移动的文件列表
$filesToMove = @(
    "create_structure.bat",
    "create_structure.ps1",
    "move_files.bat",
    "move_files.ps1",
    "cleanup.ps1"
)

# 遍历并移动文件
foreach ($file in $filesToMove) {
    if (Test-Path -Path $file) {
        Move-Item -Path $file -Destination "$scriptsDir/$file" -Force
        Write-Host "已移动到scripts目录: $file" -ForegroundColor Green
    }
    else {
        Write-Host "文件不存在: $file" -ForegroundColor Yellow
    }
}

# 添加README文件到scripts目录
$readmeContent = @"
# 实用脚本

此目录包含用于项目管理和维护的工具脚本。

## 脚本清单

- `create_structure.bat/ps1`: 创建项目目录结构
- `move_files.bat/ps1`: 移动文件到正确目录
- `cleanup.ps1`: 清理临时文件和脚本

## 使用方法

在命令行中运行这些脚本：

PowerShell: `.\scripts\script_name.ps1`
Windows命令行: `scripts\script_name.bat`
"@

Set-Content -Path "$scriptsDir/README.md" -Value $readmeContent
Write-Host "已创建脚本目录README文件" -ForegroundColor Green

Write-Host "文件整理完成!" -ForegroundColor Cyan
