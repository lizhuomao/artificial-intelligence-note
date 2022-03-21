git init 初始化仓库

git status 查看当前仓库信息

git add <filename> 将文件加入暂存区

git add . 加入所有文件

git commit -m "file discription"  提交 -m 表示message

gir log 查看日志

commit 之前可以使用 git reset <filename/hashcode> 将绿色文件变为红色 

--hard 不保存所有变更

--soft 保留变更且变更内容出于Staged

--mixed 保留变更且变更内容出于Modified

git checkout -b <name>origin<template> 分支名  以那个分支为模板  加origin 表示来自远程仓库

git branch 查看所有分支

git merge <branchName> 合并分支

git remote add origin git@github.com:yourName/yourRepo.git 关联仓库

git push origin main 把本地库的内容推送到远程库

