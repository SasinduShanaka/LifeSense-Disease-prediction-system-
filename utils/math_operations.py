"""
Mathematical Operations and Calculations
Advanced mathematical utilities and algorithms
"""
import math
import random
from decimal import Decimal, getcontext


class Calculator:
    """Advanced calculator with various operations"""
    
    @staticmethod
    def factorial(n):
        """Calculate factorial"""
        if n < 0:
            raise ValueError("Factorial not defined for negative numbers")
        return math.factorial(n)
    
    @staticmethod
    def fibonacci(n):
        """Generate Fibonacci sequence up to n terms"""
        if n <= 0:
            return []
        elif n == 1:
            return [0]
        
        fib = [0, 1]
        for i in range(2, n):
            fib.append(fib[i-1] + fib[i-2])
        
        return fib
    
    @staticmethod
    def is_prime(n):
        """Check if number is prime"""
        if n < 2:
            return False
        if n == 2:
            return True
        if n % 2 == 0:
            return False
        
        for i in range(3, int(math.sqrt(n)) + 1, 2):
            if n % i == 0:
                return False
        
        return True
    
    @staticmethod
    def prime_factors(n):
        """Find prime factors of number"""
        factors = []
        d = 2
        
        while n > 1:
            while n % d == 0:
                factors.append(d)
                n //= d
            d += 1
            if d * d > n and n > 1:
                factors.append(n)
                break
        
        return factors
    
    @staticmethod
    def gcd(a, b):
        """Greatest common divisor"""
        return math.gcd(a, b)
    
    @staticmethod
    def lcm(a, b):
        """Least common multiple"""
        return abs(a * b) // math.gcd(a, b)
    
    @staticmethod
    def power(base, exponent):
        """Calculate power"""
        return base ** exponent
    
    @staticmethod
    def square_root(n):
        """Calculate square root"""
        return math.sqrt(n)
    
    @staticmethod
    def cube_root(n):
        """Calculate cube root"""
        return n ** (1/3)
    
    @staticmethod
    def nth_root(n, root):
        """Calculate nth root"""
        return n ** (1/root)
    
    @staticmethod
    def logarithm(n, base=math.e):
        """Calculate logarithm"""
        if base == math.e:
            return math.log(n)
        return math.log(n, base)
    
    @staticmethod
    def percentage(part, whole):
        """Calculate percentage"""
        return (part / whole) * 100
    
    @staticmethod
    def compound_interest(principal, rate, time, n=1):
        """
        Calculate compound interest
        
        Args:
            principal: Initial amount
            rate: Annual interest rate (decimal)
            time: Time in years
            n: Number of times interest compounds per year
        """
        amount = principal * (1 + rate/n) ** (n * time)
        interest = amount - principal
        return {'amount': amount, 'interest': interest}


class Statistics:
    """Statistical calculations"""
    
    @staticmethod
    def mean(data):
        """Calculate arithmetic mean"""
        return sum(data) / len(data)
    
    @staticmethod
    def median(data):
        """Calculate median"""
        sorted_data = sorted(data)
        n = len(sorted_data)
        
        if n % 2 == 0:
            return (sorted_data[n//2 - 1] + sorted_data[n//2]) / 2
        return sorted_data[n//2]
    
    @staticmethod
    def mode(data):
        """Calculate mode"""
        from collections import Counter
        counts = Counter(data)
        max_count = max(counts.values())
        return [k for k, v in counts.items() if v == max_count]
    
    @staticmethod
    def variance(data):
        """Calculate variance"""
        mean = Statistics.mean(data)
        return sum((x - mean) ** 2 for x in data) / len(data)
    
    @staticmethod
    def standard_deviation(data):
        """Calculate standard deviation"""
        return math.sqrt(Statistics.variance(data))
    
    @staticmethod
    def percentile(data, p):
        """Calculate percentile"""
        sorted_data = sorted(data)
        k = (len(sorted_data) - 1) * (p / 100)
        f = math.floor(k)
        c = math.ceil(k)
        
        if f == c:
            return sorted_data[int(k)]
        
        return sorted_data[int(f)] * (c - k) + sorted_data[int(c)] * (k - f)
    
    @staticmethod
    def quartiles(data):
        """Calculate quartiles"""
        return {
            'Q1': Statistics.percentile(data, 25),
            'Q2': Statistics.median(data),
            'Q3': Statistics.percentile(data, 75)
        }
    
    @staticmethod
    def correlation(x, y):
        """Calculate Pearson correlation coefficient"""
        n = len(x)
        
        mean_x = Statistics.mean(x)
        mean_y = Statistics.mean(y)
        
        numerator = sum((x[i] - mean_x) * (y[i] - mean_y) for i in range(n))
        denominator = math.sqrt(
            sum((x[i] - mean_x) ** 2 for i in range(n)) *
            sum((y[i] - mean_y) ** 2 for i in range(n))
        )
        
        return numerator / denominator if denominator != 0 else 0


class Geometry:
    """Geometric calculations"""
    
    @staticmethod
    def circle_area(radius):
        """Calculate circle area"""
        return math.pi * radius ** 2
    
    @staticmethod
    def circle_circumference(radius):
        """Calculate circle circumference"""
        return 2 * math.pi * radius
    
    @staticmethod
    def rectangle_area(length, width):
        """Calculate rectangle area"""
        return length * width
    
    @staticmethod
    def rectangle_perimeter(length, width):
        """Calculate rectangle perimeter"""
        return 2 * (length + width)
    
    @staticmethod
    def triangle_area(base, height):
        """Calculate triangle area"""
        return 0.5 * base * height
    
    @staticmethod
    def triangle_area_heron(a, b, c):
        """Calculate triangle area using Heron's formula"""
        s = (a + b + c) / 2
        return math.sqrt(s * (s - a) * (s - b) * (s - c))
    
    @staticmethod
    def sphere_volume(radius):
        """Calculate sphere volume"""
        return (4/3) * math.pi * radius ** 3
    
    @staticmethod
    def sphere_surface_area(radius):
        """Calculate sphere surface area"""
        return 4 * math.pi * radius ** 2
    
    @staticmethod
    def cylinder_volume(radius, height):
        """Calculate cylinder volume"""
        return math.pi * radius ** 2 * height
    
    @staticmethod
    def cone_volume(radius, height):
        """Calculate cone volume"""
        return (1/3) * math.pi * radius ** 2 * height
    
    @staticmethod
    def distance_2d(x1, y1, x2, y2):
        """Calculate distance between two 2D points"""
        return math.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)
    
    @staticmethod
    def distance_3d(x1, y1, z1, x2, y2, z2):
        """Calculate distance between two 3D points"""
        return math.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2 + (z2 - z1) ** 2)


class NumberTheory:
    """Number theory operations"""
    
    @staticmethod
    def is_even(n):
        """Check if number is even"""
        return n % 2 == 0
    
    @staticmethod
    def is_odd(n):
        """Check if number is odd"""
        return n % 2 != 0
    
    @staticmethod
    def is_perfect_square(n):
        """Check if number is perfect square"""
        root = int(math.sqrt(n))
        return root * root == n
    
    @staticmethod
    def sum_of_digits(n):
        """Calculate sum of digits"""
        return sum(int(digit) for digit in str(abs(n)))
    
    @staticmethod
    def reverse_number(n):
        """Reverse digits of number"""
        return int(str(abs(n))[::-1]) * (1 if n >= 0 else -1)
    
    @staticmethod
    def is_palindrome(n):
        """Check if number is palindrome"""
        s = str(abs(n))
        return s == s[::-1]
    
    @staticmethod
    def is_armstrong(n):
        """Check if number is Armstrong number"""
        digits = [int(d) for d in str(n)]
        power = len(digits)
        return sum(d ** power for d in digits) == n
    
    @staticmethod
    def collatz_sequence(n):
        """Generate Collatz sequence"""
        sequence = [n]
        
        while n != 1:
            if n % 2 == 0:
                n = n // 2
            else:
                n = 3 * n + 1
            sequence.append(n)
        
        return sequence


def generate_random_numbers(count, min_val=0, max_val=100):
    """Generate list of random numbers"""
    return [random.randint(min_val, max_val) for _ in range(count)]


def precision_calculation(expression, decimal_places=10):
    """Calculate with high precision"""
    getcontext().prec = decimal_places
    return Decimal(expression)
