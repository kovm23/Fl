import React, { useState, useCallback, useRef, useEffect } from "react";
import {
  UploadCloud,
  FileText,
  Image as ImageIcon,
  Video,
  X,
  Check,
  AlertCircle,
  PlayCircle,
  Loader2,
  Cpu,
  Server,
  Download,
  Terminal,
  ChevronRight,
  Maximize2,
  Code
} from "lucide-react";
import { Button } from "@/components/ui/button";
import { Card, CardContent, CardHeader, CardTitle, CardDescription } from "@/components/ui/card";
import { Alert, AlertDescription, AlertTitle } from "@/components/ui/alert";
import { Badge } from "@/components/ui/badge";
import { Label } from "@/components/ui/label";
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs";
import { ScrollArea } from "@/components/ui/scroll-area";
import { Separator } from "@/components/ui/separator";

// --- KONFIGURACE MODELŮ PRO TVŮJ SERVER (40GB VRAM) ---
const AVAILABLE_MODELS = [
  { id: "gpt-4o", name: "Azure OpenAI (GPT-4o) - Cloud", type: "cloud" },
  { id: "llava:34b", name: "Llava v1.6 34B (Local - High Quality)", type: "local" },
  { id: "llama3.2-vision", name: "Llama 3.2 Vision (Local - Balanced)", type: "local" },
  { id: "llava:7b", name: "Llava 7B (Local - Fast)", type: "local" },
];

export default function MediaFeatureLabPro() {
  const [files, setFiles] = useState<File[]>([]);
  const [description, setDescription] = useState("");
  // Defaultně vybereme první model (GPT-4o), uživatel si může přepnout
  const [modelProvider, setModelProvider] = useState<string>(AVAILABLE_MODELS[0].id);
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [result, setResult] = useState<any>(null);
  const [formats, setFormats] = useState({
    json: true,
    csv: false,
    xlsx: false,
    xml: false,
  });
  const [step, setStep] = useState(1);
  const [deluxe, setDeluxe] = useState(true); // "Pro" dark mode toggle

  const fileInputRef = useRef<HTMLInputElement>(null);

  // File type detection logic
  const detectFileType = (files: File[]) => {
    if (!files.length) return null;
    const ext = files[0].name.split(".").pop()?.toLowerCase();
    if (["jpg", "jpeg", "png"].includes(ext || "")) return "image";
    if (["mp4", "avi", "mov", "mkv"].includes(ext || "")) return "video";
    if (["pdf", "txt", "md"].includes(ext || "")) return "text";
    if (["zip"].includes(ext || "")) return "zip";
    return "unknown";
  };

  const onDrop = useCallback((e: React.DragEvent) => {
    e.preventDefault();
    if (e.dataTransfer.files && e.dataTransfer.files.length > 0) {
      handleFiles(Array.from(e.dataTransfer.files));
    }
  }, []);

  const handleFiles = (newFiles: File[]) => {
    setError(null);
    setResult(null);
    
    // Validate single type
    const firstType = detectFileType([newFiles[0]]);
    const allSame = newFiles.every(f => detectFileType([f]) === firstType);
    
    if (!allSame) {
      setError("Please upload files of the same type (e.g. only images or only PDFs).");
      return;
    }
    
    setFiles(newFiles);
    setStep(2);
  };

  const removeFile = (index: number) => {
    const newFiles = [...files];
    newFiles.splice(index, 1);
    setFiles(newFiles);
    if (newFiles.length === 0) setStep(1);
  };

  // --- ODESLÁNÍ DAT NA BACKEND ---
  const handleUpload = async () => {
    if (!files.length) return;
    setBusy(true);
    setError(null);
    setResult(null);

    const formData = new FormData();
    files.forEach((f) => formData.append("files", f));

    // Formátování výstupu
    const outputFormatsString = Object.entries(formats)
      .filter(([_, v]) => v)
      .map(([k]) => k)
      .join(",");
      
    if (outputFormatsString) formData.append("output_formats", outputFormatsString);
    if (description.trim()) formData.append("description", description.trim());
    
    // DŮLEŽITÉ: Posíláme klíč "model" s ID modelu (např. "llava:34b")
    formData.append("model", modelProvider);

    const fileType = detectFileType(files) || "unknown";

    try {
      // Simulace progressu pro lepší UX
      if (fileType === 'video' && modelProvider.includes('llava')) {
         // Pokud běží lokální video analýza, může to chvíli trvat
      }

      const res = await fetch("/upload", {
        method: "POST",
        body: formData,
      });

      if (!res.ok) {
        const errData = await res.json();
        throw new Error(errData.error || "Upload failed");
      }

      const data = await res.json();
      setResult(data);
      setStep(3);
    } catch (err: any) {
      setError(err.message);
    } finally {
      setBusy(false);
    }
  };

  // Helper pro zobrazení JSONu
  const renderJsonPreview = (data: any) => {
    if (!data) return null;
    return (
      <pre className={`text-[11px] font-mono leading-relaxed p-4 rounded-lg overflow-auto max-h-[500px] ${deluxe ? "bg-[#0d1117] text-gray-300" : "bg-slate-50 text-slate-700"}`}>
        {JSON.stringify(data, null, 2)}
      </pre>
    );
  };
  
  const fileType = detectFileType(files);

  return (
    <div className={`min-h-screen transition-colors duration-500 font-sans selection:bg-indigo-500/30 ${deluxe ? "bg-[#09090b] text-slate-200" : "bg-slate-50 text-slate-900"}`}>
      
      {/* Header / Navbar */}
      <header className={`sticky top-0 z-50 border-b backdrop-blur-xl ${deluxe ? "border-white/5 bg-[#09090b]/80" : "border-slate-200 bg-white/80"}`}>
        <div className="container mx-auto px-4 h-16 flex items-center justify-between">
          <div className="flex items-center gap-3">
            <div className={`p-2 rounded-xl ${deluxe ? "bg-indigo-500/10" : "bg-indigo-600/10"}`}>
              <Cpu className={`h-5 w-5 ${deluxe ? "text-indigo-400" : "text-indigo-600"}`} />
            </div>
            <div>
              <h1 className="font-bold text-lg tracking-tight">MediaFeature<span className="text-indigo-500">Lab</span></h1>
              <p className="text-[10px] uppercase tracking-widest opacity-50 font-medium">Multimodal Processing Unit</p>
            </div>
          </div>
          
          <div className="flex items-center gap-4">
             <div className="hidden md:flex items-center gap-2 px-3 py-1.5 rounded-full border border-dashed border-slate-700/50 bg-slate-900/50">
                <div className="w-2 h-2 rounded-full bg-green-500 animate-pulse shadow-[0_0_8px_rgba(34,197,94,0.5)]"></div>
                <span className="text-[10px] font-mono text-slate-400">GPU ACTIVE: 40GB VRAM</span>
             </div>
             <Button 
               variant="ghost" 
               size="sm" 
               onClick={() => setDeluxe(!deluxe)}
               className="rounded-full w-9 h-9 p-0"
             >
               {deluxe ? "🌙" : "☀️"}
             </Button>
          </div>
        </div>
      </header>

      <main className="container mx-auto px-4 py-8 max-w-6xl">
        
        {/* Main Grid */}
        <div className="grid grid-cols-1 lg:grid-cols-12 gap-8">
          
          {/* Left Column: Input & Config */}
          <div className="lg:col-span-5 space-y-6">
            
            {/* Step 1: Upload Area */}
            <div 
              className={`relative group rounded-3xl border-2 border-dashed transition-all duration-300 overflow-hidden
                ${busy ? "pointer-events-none opacity-50" : ""}
                ${files.length > 0 
                  ? (deluxe ? "border-indigo-500/50 bg-indigo-500/5" : "border-indigo-200 bg-indigo-50/50") 
                  : (deluxe ? "border-slate-800 hover:border-slate-700 bg-slate-900/50" : "border-slate-300 hover:border-indigo-400 bg-white")
                }
              `}
              onDragOver={(e) => e.preventDefault()}
              onDrop={onDrop}
            >
              <input
                type="file"
                multiple
                className="hidden"
                ref={fileInputRef}
                onChange={(e) => e.target.files && handleFiles(Array.from(e.target.files))}
              />
              
              <div className="p-8 min-h-[320px] flex flex-col items-center justify-center text-center">
                {files.length === 0 ? (
                  <>
                     <div className={`w-20 h-20 mb-6 rounded-2xl flex items-center justify-center transition-transform group-hover:scale-110 duration-500 ${deluxe ? "bg-slate-800" : "bg-indigo-50"}`}>
                        <UploadCloud className={`h-10 w-10 ${deluxe ? "text-slate-400" : "text-indigo-500"}`} />
                     </div>
                     <h3 className="text-xl font-semibold mb-2">Drop files to analyze</h3>
                     <p className="text-sm text-slate-500 mb-8 max-w-[260px] leading-relaxed">
                       Support for Images (PNG, JPG), Videos (MP4, MKV), Docs (PDF, TXT) or ZIP archives.
                     </p>
                     <Button 
                       onClick={() => fileInputRef.current?.click()} 
                       className={`rounded-full px-8 h-12 font-medium shadow-lg hover:shadow-indigo-500/25 transition-all
                         ${deluxe ? "bg-white text-black hover:bg-slate-200" : "bg-indigo-600 text-white hover:bg-indigo-700"}`}
                     >
                       Browse Files
                     </Button>
                  </>
                ) : (
                  <div className="w-full h-full flex flex-col">
                    <div className="flex items-center justify-between mb-4 px-2">
                       <span className="text-xs font-bold uppercase tracking-wider text-indigo-500">
                         {files.length} Files Queued
                       </span>
                       <Button variant="ghost" size="sm" onClick={() => {setFiles([]); setStep(1);}} className="h-6 text-xs text-red-400 hover:text-red-300 hover:bg-red-900/20">
                         Clear All
                       </Button>
                    </div>
                    
                    <ScrollArea className="flex-1 h-[240px] w-full pr-4">
                      <div className="space-y-2">
                        {files.map((f, i) => (
                          <div key={i} className={`flex items-center gap-3 p-3 rounded-xl border group/item transition-all ${deluxe ? "bg-slate-800/50 border-slate-700/50" : "bg-white border-slate-200 shadow-sm"}`}>
                            <div className={`w-10 h-10 rounded-lg flex items-center justify-center ${deluxe ? "bg-slate-900" : "bg-slate-100 text-slate-500"}`}>
                               {detectFileType([f]) === 'image' ? <ImageIcon className="h-5 w-5" /> : 
                                detectFileType([f]) === 'video' ? <Video className="h-5 w-5" /> :
                                <FileText className="h-5 w-5" />}
                            </div>
                            <div className="flex-1 min-w-0 text-left">
                              <p className="text-sm font-medium truncate">{f.name}</p>
                              <p className="text-[10px] text-slate-500 uppercase">{(f.size / 1024 / 1024).toFixed(2)} MB</p>
                            </div>
                            <button onClick={(e) => {e.stopPropagation(); removeFile(i);}} className="opacity-0 group-hover/item:opacity-100 p-2 hover:bg-red-500/10 rounded-full text-slate-500 hover:text-red-500 transition-all">
                              <X className="h-4 w-4" />
                            </button>
                          </div>
                        ))}
                      </div>
                    </ScrollArea>
                  </div>
                )}
              </div>
            </div>

            {/* Step 2: Controls */}
            <div className={`rounded-3xl p-6 border transition-all duration-500 ${deluxe ? "bg-[#0c0f14] border-slate-800" : "bg-white border-slate-200 shadow-xl shadow-slate-200/50"} ${step < 2 ? "opacity-50 grayscale pointer-events-none blur-[1px]" : "opacity-100"}`}>
                
                <div className="flex items-center justify-between mb-6">
                  <h3 className="font-semibold flex items-center gap-2">
                    <Terminal className="h-4 w-4 text-indigo-500" />
                    Extraction Config
                  </h3>
                  <div className="px-2 py-1 rounded text-[10px] font-bold bg-indigo-500/10 text-indigo-500 border border-indigo-500/20">
                    STEP 02
                  </div>
                </div>

                <div className="space-y-5">
                   
                   {/* Description Input */}
                   <div className="space-y-2">
                     <Label className="text-xs text-slate-500 font-medium ml-1">Context Description (Optional)</Label>
                     <textarea
                        value={description}
                        onChange={(e) => setDescription(e.target.value)}
                        rows={2}
                        className={`w-full rounded-xl p-3 text-sm outline-none transition-all resize-none focus:ring-2 focus:ring-indigo-500/50
                          ${deluxe ? "bg-slate-900/50 border border-slate-700 focus:border-indigo-500 text-slate-200 placeholder:text-slate-600" : "bg-slate-50 border border-slate-200 text-slate-800"}`}
                        placeholder="Describe the dataset context (e.g. 'Medical X-Rays of chest', 'Traffic surveillance footage')..."
                      />
                   </div>

                   {/* Output Formats */}
                   <div>
                     <Label className="text-xs text-slate-500 font-medium ml-1 mb-2 block">Export Formats</Label>
                     <div className="flex gap-2">
                        {(["json", "csv", "xlsx", "xml"] as const).map((fmt) => (
                          <label key={fmt} className={`flex-1 cursor-pointer group relative overflow-hidden rounded-xl border transition-all
                            ${formats[fmt] 
                              ? (deluxe ? "bg-indigo-500/20 border-indigo-500/50 text-indigo-300" : "bg-indigo-50 border-indigo-200 text-indigo-700") 
                              : (deluxe ? "bg-slate-900 border-slate-800 text-slate-500 hover:border-slate-700" : "bg-white border-slate-200 text-slate-500 hover:border-slate-300")
                            }`}>
                             <div className="flex items-center justify-center h-10 text-xs font-bold uppercase">
                                {fmt}
                                {formats[fmt] && <Check className="h-3 w-3 ml-1.5" />}
                             </div>
                             <input type="checkbox" className="hidden" checked={formats[fmt]} onChange={(e) => setFormats(s => ({...s, [fmt]: e.target.checked}))} />
                          </label>
                        ))}
                     </div>
                   </div>

                   <Separator className={deluxe ? "bg-slate-800" : "bg-slate-100"} />

                   {/* Model Selection & Action Bar */}
                   <div className={`p-1.5 rounded-2xl flex items-center gap-1.5 border ${deluxe ? "bg-slate-900 border-slate-800" : "bg-slate-50 border-slate-200"}`}>
                      
                      {/* Model Selector */}
                      <div className="relative flex-1 min-w-0 group">
                        <div className="absolute left-3 top-1/2 -translate-y-1/2 pointer-events-none">
                           <Server className={`h-4 w-4 ${deluxe ? "text-slate-500" : "text-slate-400"}`} />
                        </div>
                        {/* ZDE JE ZMĚNA - Dynamické menu modelů */}
                        <select
                          value={modelProvider}
                          onChange={(e) => setModelProvider(e.target.value)}
                          className={`w-full h-11 pl-9 pr-8 text-xs font-medium appearance-none bg-transparent outline-none cursor-pointer rounded-xl transition-colors
                            ${deluxe 
                              ? "text-slate-300 hover:bg-slate-800 focus:bg-slate-800" 
                              : "text-slate-700 hover:bg-white focus:bg-white"}`}
                        >
                          {AVAILABLE_MODELS.map((m) => (
                            <option key={m.id} value={m.id} className={deluxe ? "bg-slate-900" : ""}>
                              {m.name}
                            </option>
                          ))}
                        </select>
                        
                        <div className="absolute right-3 top-1/2 -translate-y-1/2 pointer-events-none opacity-50">
                           <Code className="h-3 w-3" />
                        </div>
                      </div>

                      {/* Main Button */}
                      <Button 
                        onClick={handleUpload} 
                        disabled={!files.length || !!error || busy} 
                        className={`h-11 px-6 rounded-xl font-bold text-sm shadow-lg transition-all
                          ${busy ? "w-full" : "w-auto"}
                          ${deluxe 
                            ? "bg-indigo-500 hover:bg-indigo-400 text-white shadow-indigo-500/20" 
                            : "bg-indigo-600 hover:bg-indigo-700 text-white shadow-indigo-200"}`}
                      >
                        {busy ? (
                          <div className="flex items-center justify-center gap-2">
                            <Loader2 className="h-4 w-4 animate-spin" />
                            <span>Processing...</span>
                          </div>
                        ) : (
                          <div className="flex items-center gap-2">
                             <PlayCircle className="h-4 w-4" />
                             <span>Run Extraction</span>
                          </div>
                        )}
                      </Button>
                   </div>
                </div>
            </div>

            {/* Error Message */}
            {error && (
              <Alert variant="destructive" className="bg-red-500/10 border-red-500/20 text-red-500">
                <AlertCircle className="h-4 w-4" />
                <AlertTitle>Error</AlertTitle>
                <AlertDescription>{error}</AlertDescription>
              </Alert>
            )}

          </div>

          {/* Right Column: Results */}
          <div className="lg:col-span-7 h-full flex flex-col">
             
             {/* Empty State */}
             {!result && !busy && (
               <div className={`h-full min-h-[500px] rounded-3xl border-2 border-dashed flex flex-col items-center justify-center p-8 text-center opacity-50
                 ${deluxe ? "border-slate-800 bg-slate-900/20" : "border-slate-200 bg-slate-50/50"}`}>
                  <div className={`w-32 h-32 rounded-full mb-6 flex items-center justify-center ${deluxe ? "bg-slate-900" : "bg-white"}`}>
                     <Maximize2 className="h-12 w-12 text-slate-700 opacity-20" />
                  </div>
                  <h4 className="text-lg font-medium mb-2">Ready for Output</h4>
                  <p className="text-sm text-slate-500 max-w-xs">
                    Results from the LLM extraction process will appear here in real-time.
                  </p>
               </div>
             )}

             {/* Busy State */}
             {busy && (
               <div className={`h-full min-h-[500px] rounded-3xl border flex flex-col items-center justify-center p-8 relative overflow-hidden
                 ${deluxe ? "border-slate-800 bg-[#0c0f14]" : "border-slate-200 bg-white"}`}>
                  <div className="absolute inset-0 bg-[url('https://grainy-gradients.vercel.app/noise.svg')] opacity-20"></div>
                  <div className="relative z-10 flex flex-col items-center">
                    <Loader2 className="h-16 w-16 text-indigo-500 animate-spin mb-6" />
                    <h3 className="text-2xl font-bold mb-2">Processing Data</h3>
                    <p className="text-slate-500 mb-8">Running inference on {modelProvider.split(":")[0]}...</p>
                    
                    <div className="flex gap-3">
                       <div className="h-2 w-2 rounded-full bg-indigo-500 animate-bounce delay-0"></div>
                       <div className="h-2 w-2 rounded-full bg-indigo-500 animate-bounce delay-150"></div>
                       <div className="h-2 w-2 rounded-full bg-indigo-500 animate-bounce delay-300"></div>
                    </div>
                  </div>
               </div>
             )}

             {/* Results State */}
             {result && !busy && (
               <Card className={`h-full border-0 shadow-none bg-transparent flex flex-col`}>
                  
                  {/* Result Header */}
                  <div className="flex items-center justify-between mb-6">
                     <div>
                        <h2 className="text-2xl font-bold flex items-center gap-3">
                          Processing Complete
                          <Badge className="bg-green-500/10 text-green-500 border-green-500/20 hover:bg-green-500/20">Success</Badge>
                        </h2>
                        <p className="text-sm text-slate-500 mt-1">
                          Processed {result.files?.length || 0} files using <span className="font-mono text-indigo-400">{result.model_used || modelProvider}</span>
                        </p>
                     </div>
                     <div className="flex gap-2">
                        {/* Download Buttons Logic */}
                        {result.outputs && Object.entries(result.outputs).map(([fmt, data]: any) => (
                           <Button key={fmt} variant="outline" size="sm" className="h-9 gap-2 uppercase text-xs font-bold"
                             onClick={() => {
                               const blob = new Blob([data], {type: 'text/plain'});
                               const url = URL.createObjectURL(blob);
                               const a = document.createElement('a');
                               a.href = url;
                               a.download = `dataset.${fmt}`;
                               a.click();
                             }}
                           >
                             <Download className="h-3 w-3" /> {fmt}
                           </Button>
                        ))}
                     </div>
                  </div>

                  {/* Tabs for Data View */}
                  <Tabs defaultValue="preview" className="flex-1 flex flex-col">
                    <TabsList className={`w-full justify-start h-12 p-1 gap-1 mb-6 rounded-xl border ${deluxe ? "bg-slate-900 border-slate-800" : "bg-slate-100 border-slate-200"}`}>
                      <TabsTrigger value="preview" className="flex-1 h-full rounded-lg data-[state=active]:bg-indigo-600 data-[state=active]:text-white transition-all">Data Preview</TabsTrigger>
                      <TabsTrigger value="json" className="flex-1 h-full rounded-lg data-[state=active]:bg-indigo-600 data-[state=active]:text-white transition-all">Raw JSON</TabsTrigger>
                      {result.feature_specification && <TabsTrigger value="schema" className="flex-1 h-full rounded-lg data-[state=active]:bg-indigo-600 data-[state=active]:text-white transition-all">Schema</TabsTrigger>}
                    </TabsList>

                    <TabsContent value="preview" className="flex-1 mt-0">
                      <ScrollArea className={`h-[500px] rounded-2xl border ${deluxe ? "border-slate-800 bg-[#0c0f14]" : "border-slate-200 bg-white"}`}>
                         <div className="p-0">
                            {result.processing?.tabular_output && (
                              <div className="w-full overflow-x-auto">
                                <table className="w-full text-sm text-left">
                                  <thead className={`text-xs uppercase font-bold sticky top-0 z-10 ${deluxe ? "bg-slate-900 text-slate-400" : "bg-slate-50 text-slate-500"}`}>
                                    <tr>
                                      <th className="px-6 py-4">Filename</th>
                                      {/* Dynamické hlavičky z prvního řádku */}
                                      {Object.values(result.processing.tabular_output)[0] && 
                                        Object.keys((Object.values(result.processing.tabular_output)[0] as any).features || 
                                                    (Object.values(result.processing.tabular_output)[0] as any)
                                        ).slice(0, 5).map((k) => (
                                          <th key={k} className="px-6 py-4">{k}</th>
                                        ))
                                      }
                                    </tr>
                                  </thead>
                                  <tbody className="divide-y divide-slate-800/50">
                                     {Object.entries(result.processing.tabular_output).map(([fname, row]: any, i) => {
                                        const data = row.features || row;
                                        return (
                                          <tr key={i} className={`group hover:bg-indigo-500/5 transition-colors`}>
                                            <td className="px-6 py-4 font-mono text-xs opacity-50">{fname}</td>
                                            {Object.entries(data).slice(0, 5).map(([k, v]: any) => (
                                              <td key={k} className="px-6 py-4">
                                                {typeof v === 'object' ? JSON.stringify(v) : String(v).slice(0, 50)}
                                              </td>
                                            ))}
                                          </tr>
                                        )
                                     })}
                                  </tbody>
                                </table>
                              </div>
                            )}
                         </div>
                      </ScrollArea>
                    </TabsContent>

                    <TabsContent value="json" className="flex-1 mt-0">
                       {renderJsonPreview(result.processing?.tabular_output)}
                    </TabsContent>
                    
                    <TabsContent value="schema" className="flex-1 mt-0">
                       <div className={`p-6 rounded-2xl border h-full overflow-auto ${deluxe ? "bg-[#0c0f14] border-slate-800 text-slate-400" : "bg-white border-slate-200"}`}>
                          <p className="font-mono text-xs whitespace-pre-wrap">{result.processing?.feature_specification || result.feature_specification}</p>
                       </div>
                    </TabsContent>
                  </Tabs>

               </Card>
             )}
          </div>
        </div>
      </main>
    </div>
  );
}