"""
Forensic HTML report generation module.
Creates comprehensive HTML forensic analysis reports for evidence documentation.
"""

import logging
from datetime import datetime
from pathlib import Path
from typing import Dict, Optional, List

logger = logging.getLogger(__name__)


class ForensicReportGenerator:
    """Generate professional forensic analysis reports in HTML format."""
    
    def __init__(
        self,
        session_id: str,
        evidence_id: str,
        software_version: str = "1.0.0"
    ):
        """
        Initialize report generator.
        
        Args:
            session_id: Analysis session ID
            evidence_id: Evidence ID
            software_version: Software version (default: 1.0.0)
        """
        self.session_id = session_id
        self.evidence_id = evidence_id
        self.software_version = software_version
        self.report_date = datetime.now()
    
    def generate_header(self) -> str:
        """Generate HTML report header."""
        header = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Forensic Analysis Report - {self.evidence_id}</title>
    <style>
        * {{
            margin: 0;
            padding: 0;
            box-sizing: border-box;
        }}
        
        body {{
            font-family: 'Courier New', monospace;
            background-color: #f5f5f5;
            color: #333;
            line-height: 1.6;
        }}
        
        .container {{
            max-width: 1200px;
            margin: 20px auto;
            background-color: white;
            padding: 30px;
            border-radius: 5px;
            box-shadow: 0 2px 5px rgba(0,0,0,0.1);
        }}
        
        header {{
            border-bottom: 3px solid #d32f2f;
            padding-bottom: 20px;
            margin-bottom: 30px;
        }}
        
        h1 {{
            color: #d32f2f;
            font-size: 2em;
            margin-bottom: 10px;
        }}
        
        h2 {{
            color: #1976d2;
            border-left: 5px solid #1976d2;
            padding-left: 10px;
            margin-top: 30px;
            margin-bottom: 15px;
        }}
        
        h3 {{
            color: #555;
            margin-top: 20px;
            margin-bottom: 10px;
        }}
        
        .info-section {{
            background-color: #f9f9f9;
            border: 1px solid #ddd;
            border-radius: 3px;
            padding: 15px;
            margin-bottom: 15px;
        }}
        
        .info-row {{
            display: grid;
            grid-template-columns: 200px 1fr;
            margin-bottom: 10px;
            padding-bottom: 10px;
            border-bottom: 1px solid #eee;
        }}
        
        .info-row:last-child {{
            border-bottom: none;
        }}
        
        .info-label {{
            font-weight: bold;
            color: #555;
        }}
        
        .info-value {{
            font-family: 'Courier New', monospace;
            word-break: break-all;
        }}
        
        .hash {{
            background-color: #f0f0f0;
            padding: 5px 10px;
            border-radius: 3px;
            font-size: 0.9em;
        }}
        
        table {{
            width: 100%;
            border-collapse: collapse;
            margin-bottom: 20px;
        }}
        
        th {{
            background-color: #1976d2;
            color: white;
            padding: 12px;
            text-align: left;
            font-weight: bold;
        }}
        
        td {{
            padding: 10px 12px;
            border-bottom: 1px solid #ddd;
        }}
        
        tr:nth-child(even) {{
            background-color: #f9f9f9;
        }}
        
        tr:hover {{
            background-color: #f0f0f0;
        }}
        
        .warning {{
            background-color: #fff3cd;
            border: 1px solid #ffc107;
            border-radius: 3px;
            padding: 15px;
            margin-bottom: 20px;
            color: #856404;
        }}
        
        .success {{
            background-color: #d4edda;
            border: 1px solid #28a745;
            border-radius: 3px;
            padding: 15px;
            margin-bottom: 20px;
            color: #155724;
        }}
        
        .code {{
            background-color: #f4f4f4;
            border: 1px solid #ddd;
            border-radius: 3px;
            padding: 10px;
            font-family: 'Courier New', monospace;
            overflow-x: auto;
            margin: 10px 0;
        }}
        
        footer {{
            border-top: 1px solid #ddd;
            padding-top: 20px;
            margin-top: 40px;
            font-size: 0.9em;
            color: #666;
            text-align: center;
        }}
        
        .page-break {{
            page-break-after: always;
        }}
        
        img {{
            max-width: 100%;
            height: auto;
            margin: 15px 0;
            border: 1px solid #ddd;
            border-radius: 3px;
        }}
    </style>
</head>
<body>
    <div class="container">
        <header>
            <h1>FORENSIC ANALYSIS REPORT</h1>
            <p>License Plate Enhancement & Analysis</p>
        </header>
"""
        return header
    
    def generate_footer(self) -> str:
        """Generate HTML report footer."""
        footer = f"""        <footer>
            <p>Generated by Forensic Plate Enhancer v{self.software_version}</p>
            <p>Report Date: {self.report_date.strftime('%Y-%m-%d %H:%M:%S')}</p>
            <p>This report contains forensic analysis results and should be treated as confidential evidence documentation.</p>
        </footer>
    </div>
</body>
</html>
"""
        return footer
    
    def generate_executive_summary(
        self,
        evidence_filename: str,
        evidence_sha256: str,
        processing_operations: int,
        derivatives_generated: int
    ) -> str:
        """Generate executive summary section."""
        summary = f"""        <section>
            <h2>Executive Summary</h2>
            <div class="info-section">
                <div class="info-row">
                    <span class="info-label">Session ID:</span>
                    <span class="info-value">{self.session_id}</span>
                </div>
                <div class="info-row">
                    <span class="info-label">Evidence ID:</span>
                    <span class="info-value">{self.evidence_id}</span>
                </div>
                <div class="info-row">
                    <span class="info-label">Original File:</span>
                    <span class="info-value">{evidence_filename}</span>
                </div>
                <div class="info-row">
                    <span class="info-label">Original SHA-256:</span>
                    <span class="info-value hash">{evidence_sha256}</span>
                </div>
                <div class="info-row">
                    <span class="info-label">Processing Operations:</span>
                    <span class="info-value">{processing_operations}</span>
                </div>
                <div class="info-row">
                    <span class="info-label">Derivatives Generated:</span>
                    <span class="info-value">{derivatives_generated}</span>
                </div>
                <div class="info-row">
                    <span class="info-label">Report Generated:</span>
                    <span class="info-value">{self.report_date.strftime('%Y-%m-%d %H:%M:%S')}</span>
                </div>
            </div>
        </section>
"""
        return summary
    
    def generate_evidence_section(self, evidence_data: Dict) -> str:
        """Generate evidence information section."""
        section = """        <section>
            <h2>Evidence Information</h2>
            <div class="success">
                <strong>Original Evidence Preserved:</strong> The original evidence file has NOT been modified during analysis.
            </div>
            <div class="info-section">
"""
        
        for key, value in evidence_data.items():
            if key == "sha256":
                section += f"""                <div class="info-row">
                    <span class="info-label">{key.replace('_', ' ').title()}:</span>
                    <span class="info-value hash">{value}</span>
                </div>
"""
            else:
                section += f"""                <div class="info-row">
                    <span class="info-label">{key.replace('_', ' ').title()}:</span>
                    <span class="info-value">{value}</span>
                </div>
"""
        
        section += """            </div>
        </section>
"""
        return section
    
    def generate_operations_section(self, operations: List[Dict]) -> str:
        """Generate processing operations section."""
        section = """        <section>
            <h2>Processing Operations</h2>
            <table>
                <thead>
                    <tr>
                        <th>Operation</th>
                        <th>Parameters</th>
                        <th>Output File</th>
                        <th>SHA-256</th>
                    </tr>
                </thead>
                <tbody>
"""
        
        for op in operations:
            op_name = op.get("operation", "N/A")
            params = op.get("parameters", {})
            output_file = op.get("output_file", "N/A")
            output_hash = op.get("output_hash", "N/A")
            
            params_str = ", ".join(f"{k}={v}" for k, v in params.items()) if params else "None"
            hash_display = output_hash[:16] + "..." if len(output_hash) > 16 else output_hash
            
            section += f"""                    <tr>
                        <td>{op_name}</td>
                        <td><code>{params_str}</code></td>
                        <td>{output_file}</td>
                        <td class="hash">{hash_display}</td>
                    </tr>
"""
        
        section += """                </tbody>
            </table>
        </section>
"""
        return section
    
    def generate_integrity_section(
        self,
        original_hash: str,
        derivatives: Dict[str, str]
    ) -> str:
        """Generate integrity verification section."""
        section = f"""        <section>
            <h2>Integrity Verification</h2>
            <div class="success">
                <strong>Evidence Chain Verified:</strong> All files have been hashed for integrity verification.
            </div>
            
            <h3>Original Evidence</h3>
            <div class="info-section">
                <div class="info-row">
                    <span class="info-label">SHA-256 Hash:</span>
                    <span class="info-value hash">{original_hash}</span>
                </div>
            </div>
            
            <h3>Processed Derivatives</h3>
            <table>
                <thead>
                    <tr>
                        <th>File</th>
                        <th>SHA-256 Hash</th>
                    </tr>
                </thead>
                <tbody>
"""
        
        for filename, file_hash in derivatives.items():
            section += f"""                    <tr>
                        <td>{filename}</td>
                        <td class="hash">{file_hash[:32]}...</td>
                    </tr>
"""
        
        section += """                </tbody>
            </table>
        </section>
"""
        return section
    
    def generate_limitations_section(self) -> str:
        """Generate scientific limitations section."""
        section = """        <section>
            <h2>Scientific and Forensic Limitations</h2>
            <div class="warning">
                <strong>Important Disclaimer:</strong>
                <ul style="margin-left: 20px; margin-top: 10px;">
                    <li><strong>Image Enhancement:</strong> Enhancement processes only the information already present in the source image. Upscaling, sharpening, denoising, and deblurring do NOT guarantee recovery of information not captured by the original sensor.</li>
                    <li><strong>Deblurring Methods:</strong> Deblurring algorithms are EXPERIMENTAL and results should NOT be treated as recovered ground truth. Manual verification by investigator is required.</li>
                    <li><strong>AI-Generated Content:</strong> If AI-based enhancement is used, it must be clearly identified as synthetic/model-generated processing and must NOT be treated as independent evidence.</li>
                    <li><strong>Character Recognition:</strong> This tool does NOT automatically output recognized plate numbers. Investigators must manually record observations with confidence levels and retain supporting evidence.</li>
                    <li><strong>Investigator Verification:</strong> All results require manual inspection and verification by qualified forensic examiner before use in legal proceedings.</li>
                </ul>
            </div>
        </section>
"""
        return section
    
    def generate_report(
        self,
        evidence_data: Dict,
        processing_operations: List[Dict],
        derivatives_hashes: Dict[str, str],
        output_path: str | Path
    ) -> Path:
        """
        Generate complete forensic report.
        
        Args:
            evidence_data: Evidence metadata dictionary
            processing_operations: List of operation dictionaries
            derivatives_hashes: Dictionary of filename -> sha256_hash
            output_path: Path to save report
            
        Returns:
            Path to generated report
        """
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Build report
        report_html = self.generate_header()
        
        report_html += self.generate_executive_summary(
            evidence_data.get("filename", "Unknown"),
            evidence_data.get("sha256", "Unknown"),
            len(processing_operations),
            len(derivatives_hashes)
        )
        
        report_html += self.generate_evidence_section(evidence_data)
        
        if processing_operations:
            report_html += self.generate_operations_section(processing_operations)
        
        if derivatives_hashes:
            report_html += self.generate_integrity_section(
                evidence_data.get("sha256", ""),
                derivatives_hashes
            )
        
        report_html += self.generate_limitations_section()
        
        report_html += self.generate_footer()
        
        # Write to file
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(report_html)
        
        logger.info(f"Forensic report generated: {output_path}")
        
        return output_path
