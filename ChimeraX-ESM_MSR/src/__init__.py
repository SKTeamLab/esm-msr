# vim: set expandtab shiftwidth=4 softtabstop=4:
from chimerax.core.toolshed import BundleAPI


class _ESM_MSR_API(BundleAPI):
    api_version = 1

    @staticmethod
    def get_class(class_name):
        if class_name == "ESM_MSR":
            from . import tool
            return tool.ESM_MSR_Tool
        return None

    @staticmethod
    def start_tool(session, bundle_info_obj, tool_info_obj):
        tool_class = _ESM_MSR_API.get_class(tool_info_obj.name)
        if tool_class is not None:
            return tool_class(session, tool_info_obj.name)
        session.logger.error(f"Failed to get class for tool: {tool_info_obj.name}")
        return None


bundle_api = _ESM_MSR_API()
