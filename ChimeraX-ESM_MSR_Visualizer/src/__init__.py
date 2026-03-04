# vim: set expandtab shiftwidth=4 softtabstop=4:
from chimerax.core.toolshed import BundleAPI

print("****** ESM_MSR_Visualizer/__init__.py TOP LEVEL EXECUTING ******")

class _ESM_MSR_VisualizerAPI(BundleAPI):
    api_version = 1
    print("****** _ESM_MSR_VisualizerAPI class DEFINED ******")

    @staticmethod
    def get_class(class_name):
        print(f"****** RSV_API.get_class called with: {class_name} ******")
        if class_name == "ESM_MSR_Visualizer":
            from . import tool
            return tool.ESM_MSR_VisualizerTool
        return None

    @staticmethod
    def start_tool(session, bundle_info_obj, tool_info_obj):
        print(f"****** RSV_API.start_tool CALLED for tool: {tool_info_obj.name} ******")
        tool_class = _ESM_MSR_VisualizerAPI.get_class(tool_info_obj.name)
        if tool_class:
            instance = tool_class(session, tool_info_obj.name)
            print(f"****** RSV_API.start_tool: Instantiated {tool_info_obj.name}, returning instance. ******")
            return instance
        session.logger.error(f"Failed to get class for tool: {tool_info_obj.name}")
        return None

    @staticmethod
    def bundle_info():
        print("****** RSV_API.bundle_info called ******")
        return {
            'name': 'ESM_MSR_VisualizerBundleAPI',
            'version': '0.1.0',
            'apis': {
                'tools': [{
                    'class': 'ESM_MSR_Visualizer', # Name passed to tool_info_obj.name and get_class
                    'name': 'Residue Score Visualizer', # Display name in Tools menu
                    'description': 'Visualizes scores from CSV on residues.',
                }],
            },
        }

print("****** Creating bundle_api instance of _ESM_MSR_VisualizerAPI ******")
bundle_api = _ESM_MSR_VisualizerAPI()
print("****** bundle_api instance CREATED ******")