import pickle
import tldextract
import argparse
import json
from gib import gib_detect_train
from dga_routines import count_consonants, entropy


def read_file(filename):
    """Generator to read lines from a file."""
    with open(filename) as f:
        for line in f:
            yield line.strip("\n")


def domain_check(domain):
    """Check and analyze a domain's characteristics."""
    # Skip Tor domains
    if domain.endswith(".onion"):
        print("Tor domains are ignored...")
        return None

    # Extract main domain name without subdomain and TLD
    domain_without_sub = tldextract.extract(domain).domain

    # Skip localized and short domains
    if domain_without_sub.startswith("xn-"):
        print("Localized domains are ignored...")
        return None
    if len(domain_without_sub) < 6:
        print("Short domains are ignored...")
        return None

    # Compute domain characteristics
    domain_entropy = entropy(domain_without_sub)
    domain_consonants = count_consonants(domain_without_sub)
    domain_length = len(domain_without_sub)

    return {
        "domain": domain_without_sub,
        "entropy": domain_entropy,
        "consonants": domain_consonants,
        "length": domain_length,
    }


def analyze_domain(domain_data, domain, model_mat, threshold):
    """Analyze a single domain using given metrics and DGA model."""
    if not domain_data:
        return None

    print(f"Analyzing domain: {domain}...")
    high_entropy = domain_data["entropy"] > 3.8
    high_consonants = domain_data["consonants"] > 7
    long_name = domain_data["length"] > 12

    if high_entropy:
        print(f"High entropy (>3.8) is a strong indicator of DGA domain.\n"
              f"This domain scored: {domain_data['entropy']}")
    if high_consonants:
        print(f"High consonants (>7) count is an indicator of DGA domain.\n"
              f"This domain scored: {domain_data['consonants']}")
    if long_name:
        print(f"Long domain name (>12) can also indicate DGA.\n"
              f"This domain scored: {domain_data['length']}")

    is_dga = gib_detect_train.avg_transition_prob(domain_data["domain"], model_mat) <= threshold

    if is_dga:
        print(f"Domain {domain} is DGA!")
    else:
        print(f"Domain {domain} is not DGA! Probably safe.\n"
              f"Additional information:\n"
              f"Entropy: {domain_data['entropy']}\n"
              f"Consonants count: {domain_data['consonants']}\n"
              f"Name length: {domain_data['length']}")

    return {
        "domain": domain,
        "is_dga": is_dga,
        "high_entropy": domain_data["entropy"] if high_entropy else None,
        "high_consonants": domain_data["consonants"] if high_consonants else None,
        "long_domain": domain_data["length"] if long_name else None,
    }


def main():
    """Main function to handle domain analysis."""
    parser = argparse.ArgumentParser(description="DGA domain detection")
    parser.add_argument("-d", "--domain", help="Domain to check")
    parser.add_argument("-f", "--file", help="File with domains (one per line)")
    args = parser.parse_args()

    # Load the model
    with open('gib/gib_model.pki', 'rb') as model_file:
        model_data = pickle.load(model_file)
    model_mat = model_data['mat']
    threshold = model_data['thresh']

    # Analyze a single domain
    if args.domain:
        domain_data = domain_check(args.domain)
        analyze_domain(domain_data, args.domain, model_mat, threshold)

    # Analyze multiple domains from a file
    elif args.file:
        results = []
        for domain in read_file(args.file):
            print(f"Processing domain: {domain}")
            domain_data = domain_check(domain)
            result = analyze_domain(domain_data, domain, model_mat, threshold)
            if result:
                results.append(result)

        # Save results to JSON file
        with open("dga_domains.json", "w") as output_file:
            json.dump(results, output_file, indent=4)
        print("File dga_domains.json has been created.")

    # Print usage if no arguments are provided
    else:
        print('''
_______________________       ________     _____           _____
___  __ \_  ____/__    |      ___  __ \______  /_____________  /______________
__  / / /  / __ __  /| |      __  / / /  _ \  __/  _ \  ___/  __/  __ \_  ___/
_  /_/ // /_/ / _  ___ |      _  /_/ //  __/ /_ /  __/ /__ / /_ / /_/ /  /
/_____/ \____/  /_/  |_|      /_____/ \___/\__/ \___/\___/ \__/ \____//_/
        ''')
        parser.print_help()


if __name__ == "__main__":
    main()
