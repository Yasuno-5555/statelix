
import os
import glob
import sys

class WasmPluginLoader:
    """
    Scans the 'plugins_wasm' directory and loads valid .wasm modules.
    Requires 'wasmtime' package. If not present, plugins are disabled.
    """
    def __init__(self, plugin_dir="plugins_wasm"):
        # Resolve absolute path relative to CWD (usually project root)
        self.plugin_dir = os.path.abspath(plugin_dir)
        self.plugins = {} # Name -> {module, functions}
        self.runtime_available = False
        self._check_runtime()

    def _check_runtime(self):
        try:
            import wasmtime
            from wasmtime import Store, Module, Instance, Func, FuncType
            self.wasm = wasmtime
            self.runtime_available = True
        except ImportError:
            print("[Statelix] 'wasmtime' not found. WASM plugins disabled.")
            self.runtime_available = False

    def scan_and_load(self):
        if not self.runtime_available:
            return {}

        if not os.path.exists(self.plugin_dir):
            os.makedirs(self.plugin_dir, exist_ok=True)
            return {}

        wasm_files = glob.glob(os.path.join(self.plugin_dir, "*.wasm"))
        loaded = {}

        store = self.wasm.Store()

        for path in wasm_files:
            filename = os.path.basename(path)
            name = os.path.splitext(filename)[0]
            
            try:
                module = self.wasm.Module.from_file(store.engine, path)
                instance = self.wasm.Instance(store, module, [])
                
                # Introspect exports
                exports = instance.exports(store)
                funcs = {}
                for export in exports:
                     # We assume plugins export functions.
                     # In a real system, we'd check types.
                     # For now, just store them.
                     funcs[export.name] = export
                
                loaded[name] = {
                    "path": path,
                    "exports": funcs,
                    "store": store # Note: Store is shared? Thread safety issues potentially.
                                   # Simple implementation: One store per session.
                }
                print(f"[Statelix] Loaded WASM Plugin: {name}")

            except Exception as e:
                print(f"[Statelix] Failed to load {name}: {e}")

        self.plugins = loaded
        return self.plugins

    def execute(self, plugin_name, func_name, *args):
        if plugin_name not in self.plugins:
            raise ValueError(f"Plugin {plugin_name} not found.")
        
        plugin = self.plugins[plugin_name]
        if func_name not in plugin['exports']:
             raise ValueError(f"Function {func_name} not found in {plugin_name}.")
             
        func = plugin['exports'][func_name]
        store = plugin['store']
        
        # Call
        return func(store, *args)
