import time
import pandas as pd
from fpdf import FPDF  # Ensure fpdf2 is installed

class ReportGenerator:
    def __init__(self, output_file=r"Report Generated\ecg_report.pdf", logo_path=r"src\logo.png"):
        self.output_file = output_file
        self.logo_path = logo_path
        self.full_report = []
        
    def add_section(self, title, content):
        """Add a section to the report"""
        self.full_report.append({
            "title": title,
            "content": content
        })
    
    def generate_report(self, patient_data, abnormalities):
        pdf = FPDF()
        pdf.add_page()
        
        # Add logo with increased size
        if self.logo_path:
            # Adjust width (w) and height (h) for the logo
            pdf.image(self.logo_path, x=45, y=0, w=50, h=50)  # Increased size to 50x50 mm
        
        pdf.set_font("Arial", style='B', size=16)
        pdf.cell(200, 10, txt="ECG Diagnostic Report", ln=1, align='C')
        pdf.ln(10)
        
        # Patient Information
        pdf.set_font("Arial", style='B', size=12)
        pdf.cell(200, 10, txt="Patient Information:", ln=1)
        pdf.set_font("Arial", size=12)
        pdf.cell(200, 10, txt=f"Name: {patient_data.get('name', 'N/A')}", ln=1)
        pdf.cell(200, 10, txt=f"Age: {patient_data.get('age', 'N/A')}", ln=1)
        pdf.cell(200, 10, txt=f"Report Date: {time.strftime('%Y-%m-%d %H:%M:%S')}", ln=1)
        pdf.ln(5)
        
        # Add analysis sections
        for section in self.full_report:
            pdf.set_font("Arial", style='B', size=12)
            pdf.cell(200, 10, txt=section["title"], ln=1)
            pdf.set_font("Arial", style='I', size=12)
            pdf.multi_cell(0, 10, txt=section["content"])
            pdf.ln(5)
            
        # Add abnormalities table
        if abnormalities:
            pdf.add_page()
            pdf.set_font("Arial", style='B', size=12)
            pdf.cell(200, 10, txt="Detected Abnormalities", ln=1)
            pdf.ln(5)
            
            col_widths = [40, 50, 40, 60]
            headers = ["Timestamp", "Condition", "Confidence", "Recommendation"]
            
            # Header
            pdf.set_fill_color(200, 220, 255)
            for i, header in enumerate(headers):
                pdf.cell(col_widths[i], 10, header, 1, 0, 'C', 1)
            pdf.ln()
            
            # Rows
            pdf.set_font("Arial", size=12)
            for entry in abnormalities:
                pdf.cell(col_widths[0], 10, entry["timestamp"], 1)
                pdf.cell(col_widths[1], 10, entry["class_name"], 1)
                pdf.cell(col_widths[2], 10, f"{entry['confidence']:.2f}", 1)
                pdf.multi_cell(col_widths[3], 10, self._get_recommendation(entry["class_name"]), 1)
                
        # Save PDF
        pdf.output(self.output_file)
        print(f"✅ Comprehensive report generated: {self.output_file}")
    
    def _get_recommendation(self, condition):
        """Get clinical recommendation based on condition"""
        recommendations = {
            "Normal": "Routine follow-up",
            "Artial Premature": "24-hour Holter monitoring recommended",
            "Premature ventricular contraction": "Cardiology consultation advised",
            "Fusion of ventricular and normal": "Urgent cardiac evaluation",
            "Fusion of paced and normal": "Device check recommended"
        }
        return recommendations.get(condition, "Clinical correlation advised")