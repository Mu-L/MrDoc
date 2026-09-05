import logging

from django.core.management.base import BaseCommand, CommandError

from app_doc.models import Project,Doc
from app_ai.util.llm_wiki import build_sections_pipeline

logger = logging.getLogger(__name__)


class Command(BaseCommand):
    help = "重建AI文档向量索引（支持 project / role / all）"

    def add_arguments(self, parser):
        parser.add_argument(
            '--projects',
            type=str,
            help='文集ID（多个用逗号分隔），或使用 all 表示全量'
        )

        parser.add_argument(
            '--roles',
            type=str,
            help='文集role（多个用逗号分隔，0公开,1私密,2指定用户可见,3访问码可见）'
        )

        parser.add_argument(
            '--dry-run',
            action='store_true',
            help='仅预览执行范围，不实际执行'
        )

    def handle(self, *args, **options):

        projects_arg = options.get('projects')
        roles_arg = options.get('roles')
        dry_run = options.get('dry_run')

        # =========================
        # 1. 获取 queryset
        # =========================
        queryset = self.get_projects_queryset(projects_arg, roles_arg)

        total = queryset.count()

        if total == 0:
            raise CommandError("没有匹配的文集")

        # =========================
        # 2. dry-run 模式
        # =========================
        if dry_run:
            self.stdout.write(self.style.WARNING("=== DRY RUN ==="))
            self.stdout.write(f"匹配文集数量: {total}")

            for p in queryset[:20]:
                self.stdout.write(f"- {p.id} | {p.name}")

            if total > 20:
                self.stdout.write(f"... 还有 {total - 20} 个未显示")

            return

        # =========================
        # 3. 全量风险提示
        # =========================
        if projects_arg == "all":
            self.stdout.write(self.style.WARNING(
                "⚠️ 即将执行【全站文集AI索引重建】"
            ))

            confirm = input("输入 YES 确认执行：")
            if confirm != "YES":
                self.stdout.write(self.style.ERROR("已取消执行"))
                return

        # =========================
        # 4. 执行
        # =========================
        self.stdout.write(self.style.SUCCESS(
            f"开始重建AI索引，共 {total} 个文集"
        ))

        result = self.rebuild_projects_index(queryset)

        # =========================
        # 5. 输出结果
        # =========================
        self.stdout.write(self.style.SUCCESS(
            "\n========== 完成 ==========\n"
            f"文集数量: {result['projects']}\n"
            f"文档数量: {result['docs']}\n"
            f"成功数量: {result['success']}\n"
            f"失败数量: {result['failed']}\n"
        ))

    def get_projects_queryset(self, projects_arg, roles_arg):

        # =========================
        # ❗安全规则：必须显式指定
        # =========================
        if not projects_arg and not roles_arg:
            raise CommandError(
                "禁止默认全量执行，请使用：\n"
                "--projects=all（全量）\n"
                "--projects=1,2,3（指定文集ID）\n"
                "--roles=0,1（指定文集权限，0公开,1私密,2指定用户可见,3访问码可见）"
            )

        # =========================
        # 1. 全量模式：全站已发布文集文档均参与索引（无范围配置）
        # =========================
        if projects_arg == "all":
            return Project.objects.all()

        # =========================
        # 2. project 优先级最高
        # =========================
        if projects_arg:

            try:
                project_ids = [
                    int(i.strip())
                    for i in projects_arg.split(',')
                    if i.strip()
                ]

            except ValueError:
                raise CommandError("projects 参数格式错误，应为 1,2,3")

            return Project.objects.filter(id__in=project_ids)

        # =========================
        # 3. role 模式
        # =========================
        if roles_arg:

            try:
                role_list = [
                    int(i.strip())
                    for i in roles_arg.split(',')
                    if i.strip()
                ]

            except ValueError:
                raise CommandError("roles 参数格式错误，应为 0,1,2,3")

            return Project.objects.filter(role__in=role_list)

        # 理论不会走到这里（防御）
        raise CommandError("参数解析失败")

    def rebuild_project_docs_index(self,project):
        """
        重建单个文集下所有文档的AI索引
        """
        docs = Doc.objects.filter(top_doc=project.id, status=1)

        total = docs.count()
        success = 0
        failed = 0

        self.stdout.write(self.style.WARNING(f"开始重建文集索引：{project.name}（ID:{project.id}）"))

        for doc in docs:
            try:
                build_sections_pipeline(doc)
                success += 1

                self.stdout.write(self.style.SUCCESS(
                    f"文档索引重建成功：DocID={doc.id} "
                    f"Title={doc.name}"
                ))

            except Exception as e:
                failed += 1

                self.stdout.write(self.style.ERROR(
                    f"文档索引重建失败：DocID={doc.id} "
                    f"Title={doc.name} "
                    f"Error={str(e)}"
                ))

        self.stdout.write(self.style.SUCCESS(
            f"文集重建完成：ProjectID={project.id} "
            f"Total={total} Success={success} Failed={failed}"
        ))

        return {
            "total": total,
            "success": success,
            "failed": failed
        }

    def rebuild_projects_index(self,projects_queryset):
        """
        批量重建多个文集索引
        """
        total_projects = 0
        total_docs = 0
        total_success = 0
        total_failed = 0

        for project in projects_queryset:
            total_projects += 1

            result = self.rebuild_project_docs_index(project)

            total_docs += result["total"]
            total_success += result["success"]
            total_failed += result["failed"]

        return {
            "projects": total_projects,
            "docs": total_docs,
            "success": total_success,
            "failed": total_failed
        }
