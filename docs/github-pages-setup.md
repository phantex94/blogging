# GitHub Pages 配置指南

根据GitHub Pages的最新文档，以下是配置GitHub Pages发布源的步骤。

## 配置GitHub Pages发布源

1. 登录GitHub，导航到您的仓库 `https://github.com/phantex94/blogging`
2. 点击仓库顶部的 **Settings** 选项卡
3. 在左侧菜单中，点击 **Pages**
4. 在 **Source** 部分，选择 **GitHub Actions** 作为发布源
   ![GitHub Pages Source Setting](https://docs.github.com/assets/cb-32892/mw-1440/images/help/pages/publishing-source-actions.webp)
5. 点击 **Save** 保存设置

## 防止Jekyll处理

GitHub Pages默认会使用Jekyll处理仓库中的内容。由于我们使用Hexo生成静态文件，不需要Jekyll的处理，因此我们在工作流中添加了一个`.nojekyll`文件，告诉GitHub Pages不要使用Jekyll处理我们的网站。

这个文件是在GitHub Actions工作流的构建步骤中自动创建的，您不需要手动添加。

## GitHub Actions工作流

本仓库已配置了GitHub Actions工作流，位于 `.github/workflows/deploy.yml`。该工作流会在每次推送到master分支时自动构建并部署Hexo博客。

工作流的主要步骤包括：

1. 检出代码
2. 设置Node.js环境
3. 缓存NPM依赖
4. 安装依赖
5. 构建Hexo静态文件
6. 创建`.nojekyll`文件
7. 上传构建产物
8. 部署到GitHub Pages

## 自定义域名（可选）

如果您想使用自定义域名，请按照以下步骤操作：

1. 在 `hexo-blog/source` 目录下创建一个名为 `CNAME` 的文件
2. 在文件中添加您的自定义域名，例如 `blog.example.com`
3. 在您的DNS提供商处添加相应的DNS记录，指向 `phantex94.github.io`
4. 推送更改到GitHub，GitHub Actions将自动部署包含CNAME文件的博客

## 故障排除

如果部署失败，请检查以下几点：

1. 确保GitHub Pages的发布源设置为 **GitHub Actions**
2. 检查GitHub Actions工作流的日志，查看是否有构建或部署错误
3. 确保Hexo配置文件 `_config.yml` 中的URL设置正确
4. 确保仓库的权限设置允许GitHub Actions写入Pages
5. 确保`.nojekyll`文件存在于生成的静态文件中，防止Jekyll处理

## 参考链接

- [GitHub Pages 文档](https://docs.github.com/en/pages/getting-started-with-github-pages/configuring-a-publishing-source-for-your-github-pages-site)
- [Hexo 部署文档](https://hexo.io/docs/github-pages)
- [GitHub Actions 文档](https://docs.github.com/en/actions) 