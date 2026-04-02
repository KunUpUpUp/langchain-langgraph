# Git 常用操作指南

## 基本操作
git init 初始化仓库，git clone 克隆远程仓库。
git add 暂存文件，git commit 提交更改，git push 推送到远程。

## 分支管理
git branch 查看分支，git checkout -b 创建并切换分支。
git merge 合并分支，合并前建议先 git pull 拉取最新代码。
发生冲突时需要手动编辑冲突文件，然后 git add 和 git commit。

## 撤销操作
git reset --soft HEAD~1 撤销最近一次提交但保留更改。
git reset --hard HEAD~1 彻底撤销最近一次提交。
git stash 临时保存工作区更改，git stash pop 恢复。

## SSH 配置
使用 ssh-keygen -t ed25519 生成密钥对。
将公钥添加到 GitHub Settings 的 SSH keys 中。
如果 22 端口被封，可以在 ~/.ssh/config 中配置走 443 端口：Host github.com, Hostname ssh.github.com, Port 443。
