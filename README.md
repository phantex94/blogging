# 个人博客

一个基于Hexo的个人博客，使用GitHub Actions自动部署到GitHub Pages。

## 部署状态

[![Deploy Hexo site to GitHub Pages](https://github.com/phantex94/blogging/actions/workflows/deploy.yml/badge.svg)](https://github.com/phantex94/blogging/actions/workflows/deploy.yml)

## 博客地址

[https://phantex94.github.io/blogging](https://phantex94.github.io/blogging)

## 本地开发

### 安装依赖

```bash
cd hexo-blog
npm install
```

### 本地预览

```bash
cd hexo-blog
npm run server
```

访问 [http://localhost:4000](http://localhost:4000) 预览博客。

### 创建新文章

```bash
cd hexo-blog
npx hexo new "文章标题"
```

## 自动部署

本项目使用GitHub Actions自动部署到GitHub Pages。每当推送到main分支时，GitHub Actions会自动构建并部署博客。

### 部署流程

1. 推送更改到main分支
2. GitHub Actions自动运行部署工作流
3. 构建Hexo静态文件
4. 部署到GitHub Pages

### GitHub Pages设置

为了使GitHub Actions能够正确部署，请确保在GitHub仓库设置中：

1. 导航到仓库的Settings > Pages
2. 在Source部分，选择"GitHub Actions"作为发布源

## 文件结构

- `hexo-blog/`: Hexo博客源文件
  - `source/_posts/`: 博客文章
  - `source/about/`: 关于页面
  - `themes/`: 博客主题
- `.github/workflows/`: GitHub Actions工作流配置

## 许可证

MIT

1. 项目概述
本项目基于 Hexo 静态博客框架，采用 Vivia 主题，托管在 GitHub Pages 上，旨在构建一个美观、简洁且适用于 桌面端和移动端 的个人博客。

2. 主要功能
博客文章管理（Markdown 格式）
代码高亮支持
响应式设计，适配桌面端 & 移动端
深色/浅色模式切换
分类与标签管理
搜索功能
文章目录（TOC）
友情链接、社交媒体集成
站点 SEO 优化（支持 Open Graph、JSON-LD）
访问统计（支持 Google Analytics）
评论系统（可选支持 Waline、Giscus、Disqus）

3. 技术栈
框架：Hexo
主题：Vivia（主题仓库）
部署：GitHub Pages / Vercel / Cloudflare Pages（可选）
Markdown 解析：Hexo + Marked
搜索：Hexo Search 插件
评论系统：
默认：Waline（轻量级、无数据库）
可选：Giscus（基于 GitHub Discussions）、Disqus