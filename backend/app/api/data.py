"""
Data Upload API endpoints
"""

from fastapi import APIRouter, Depends, HTTPException, UploadFile, File, status
from sqlalchemy.orm import Session
import pandas as pd
import io
from datetime import datetime
from typing import Dict

from app.database import get_db
from app.models.service import ServiceRecord

router = APIRouter()


@router.post("/upload")
async def upload_data(
    file: UploadFile = File(...),
    db: Session = Depends(get_db)
) -> Dict[str, any]:
    """
    Upload CSV file with service data

    Expected CSV columns:
    - Service_ID, Service_Date, Location, Branch, Region, Franchise_ID
    - Product_Category, Service_Type, Technician_ID, Customer_ID
    - Service_Duration, Part_Name, Parts_Used, Parts_Cost, Parts_Revenue
    - Service_Cost, Service_Revenue, Total_Revenue
    - Warranty_Claim, Customer_Satisfaction, Product_Age_Months
    - First_Call_Resolution, Priority
    """

    # Validate file type
    if not file.filename.endswith('.csv'):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Only CSV files are supported"
        )

    try:
        # Read CSV file
        contents = await file.read()
        df = pd.read_csv(io.StringIO(contents.decode('utf-8')))

        # Validate required columns
        required_columns = [
            'Service_ID', 'Service_Date', 'Location', 'Product_Category',
            'Service_Type', 'Total_Revenue'
        ]

        missing_columns = [col for col in required_columns if col not in df.columns]
        if missing_columns:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Missing required columns: {', '.join(missing_columns)}"
            )

        # Clear existing data (optional - you might want to append instead)
        db.query(ServiceRecord).delete()
        db.commit()

        # Insert new records
        records_added = 0
        for _, row in df.iterrows():
            try:
                # Parse date
                service_date = pd.to_datetime(row['Service_Date']).date()

                # Create service record
                record = ServiceRecord(
                    service_id=str(row['Service_ID']),
                    service_date=service_date,
                    location=str(row.get('Location', '')),
                    branch=str(row.get('Branch', '')),
                    region=str(row.get('Region', '')),
                    franchise_id=str(row.get('Franchise_ID', '')),
                    product_category=str(row.get('Product_Category', '')),
                    service_type=str(row.get('Service_Type', '')),
                    technician_id=str(row.get('Technician_ID', '')),
                    customer_id=str(row.get('Customer_ID', '')),
                    service_duration=float(row.get('Service_Duration', 0)),
                    part_name=str(row.get('Part_Name', 'None')),
                    parts_used=int(row.get('Parts_Used', 0)),
                    parts_cost=float(row.get('Parts_Cost', 0)),
                    parts_revenue=float(row.get('Parts_Revenue', 0)),
                    service_cost=float(row.get('Service_Cost', 0)),
                    service_revenue=float(row.get('Service_Revenue', 0)),
                    total_revenue=float(row.get('Total_Revenue', 0)),
                    warranty_claim=int(row.get('Warranty_Claim', 0)),
                    customer_satisfaction=float(row.get('Customer_Satisfaction', 0)),
                    product_age_months=int(row.get('Product_Age_Months', 0)),
                    first_call_resolution=int(row.get('First_Call_Resolution', 0)),
                    priority=int(row.get('Priority', 1))
                )

                db.add(record)
                records_added += 1

                # Commit in batches
                if records_added % 1000 == 0:
                    db.commit()

            except Exception as e:
                print(f"Error processing row: {e}")
                continue

        # Final commit
        db.commit()

        return {
            "message": "Data uploaded successfully",
            "records_count": records_added,
            "filename": file.filename
        }

    except pd.errors.ParserError as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Error parsing CSV file: {str(e)}"
        )
    except Exception as e:
        db.rollback()
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error uploading data: {str(e)}"
        )
