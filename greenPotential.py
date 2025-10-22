
import argparse
import logging
from src.green_pont import load_json_with_comments, antwerp_green_map, create_green_map, plot_green_map, create_special_green_map

logger = logging.getLogger()
logger.setLevel(logging.INFO)
logging.basicConfig() 


# usage example:# python greenPotential.py -o create_green_map

def main():
    
    """
        Main function to run different parts of the green potential analysis based on command line argument
        Options:
            -o antwerp_green_map : Create green potential map as done in Antwerp project
            -o create_green_map : Create green potential map based on air quality, wind comfort and heat stress maps
            -o plot_green_map : Plot the resulting green potential map
            -o create_special_green_map : Create points along building boundaries where greening could be placed if facade values above ebg thresholds and hedges could be placed if no2 values above threshold
    
        usage: python greenPotential.py -o <option>
    
    """
    
    desc = """run openFoam."""
    p = argparse.ArgumentParser(description=desc)
    p.add_argument( "-o", "--option", help='mesh or run or resart' )
    args = p.parse_args()
    
    
    opt= args.option #'run'

    cf=load_json_with_comments('etc/settings_greenPotential.json')

    if args.option == 'antwerp_green_map':
        antwerp_green_map(cf)
    elif args.option == 'create_green_map':
        create_green_map(cf, method=3, aq_weight=10, comfort_weight=1, heat_weight=1)
    elif args.option == "plot_green_map":
        plot_green_map(cf)
    elif args.option == "create_special_green_map": 
        create_special_green_map(cf)
    else:
        logger.error("Unrecognized command line argument")



if __name__ == "__main__":
        main()
    