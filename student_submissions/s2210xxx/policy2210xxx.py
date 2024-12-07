from policy import Policy

import random
import numpy as np


class Policy2210xxx(Policy):
# GA (Genetic Algorithm)
    def __init__(self, policy_id = 1, population_size = 5, generations = 20, mutation_rate = 0.05):
        assert policy_id in [1, 2], "Policy ID must be 1 or 2"

        # Student code here
        if policy_id == 1:
            """
            Khởi tạo policy sử dụng Genetic Algorithm (GA).
            :param population_size: Số lượng cá thể trong quần thể.
            :param generations: Số thế hệ (iterations) thuật toán sẽ chạy.
            :param mutation_rate: Xác suất xảy ra đột biến cho mỗi cá thể trong quần thể.
            """
            self.population_size = population_size
            self.generations = generations
            self.mutation_rate = mutation_rate

        elif policy_id == 2:
            pass

    def get_action(self, observation, info):
        # Student code here
        """
        Lấy hành động dựa trên Genetic Algorithm.
        """
        # Sắp xếp các sản phẩm theo diện tích giảm dần, để ưu tiên cắt các sản phẩm lớn hơn
        products = sorted(observation["products"], key = lambda p: p["size"][0] * p["size"][1], reverse = True)
        stocks = observation["stocks"]

        # Khởi tạo quần thể
        population = self._initialize_population(products, stocks)

        # Nếu quần thể không có cá thể nào, trả về hành động mặc định
        if not population:
            return {"stock_idx": -1, "size": (0, 0), "position": (0, 0)}

        previous_best_fitness = float("inf")

        # Tiến hành qua các thế hệ
        for generation in range(self.generations):
            # Đánh giá fitness của từng cá thể trong quần thể
            fitness_scores = [self._evaluate_solution(individual, products, stocks) for individual in population]

            # Lựa chọn các cá thể tốt nhất (có fitness cao nhất)
            population = self._select_population(population, fitness_scores)

            # Lai ghép và đột biến để tạo ra thế hệ tiếp theo
            next_generation = self._crossover_and_mutate(population, products, stocks)
            population.extend(next_generation)

            # Lưu lại fitness tốt nhất
            best_fitness = min(fitness_scores)

            # Dừng sớm nếu không có sự cải thiện đáng kể trong fitness
            if generation > 5 and abs(best_fitness - previous_best_fitness) < 1e-5:
                break
            previous_best_fitness = best_fitness

        # Chọn cá thể tốt nhất từ quần thể
        best_individual = min(population, key = lambda sol: self._evaluate_solution(sol, products, stocks))

        # Trả về kết quả của cá thể tốt nhất
        if not best_individual:
            return {"stock_idx": -1, "size": (0, 0), "position": (0, 0)}
        return best_individual[0]

    # Student code here
    # You can add more functions if needed    
    def _initialize_population(self, products, stocks):
        """
        Khởi tạo quần thể bằng cách tạo ra các cá thể ngẫu nhiên.
        """
        population = []
        # Tạo ngẫu nhiên các cá thể trong quần thể
        for _ in range(self.population_size):
            individual = []
            for product in products:
                if product["quantity"] > 0:
                    # Duyệt qua các stock để kiểm tra vị trí có thể đặt sản phẩm
                    for stock_idx, stock in enumerate(stocks):
                        stock_w, stock_h = self._get_stock_size_(stock)
                        prod_w, prod_h = product["size"]

                        # Tìm vị trí hợp lý cho sản phẩm trong stock (bao gồm việc xoay sản phẩm)
                        for x_pos in range(stock_w - prod_w + 1):
                            for y_pos in range(stock_h - prod_h + 1):
                                if self._can_place_(stock, (x_pos, y_pos), (prod_w, prod_h)):
                                    individual.append({"stock_idx": stock_idx, "size": (prod_w, prod_h), "position": (x_pos, y_pos)})
                                    break
                            if individual:
                                break

                        # Nếu không thể đặt sản phẩm theo hướng ban đầu, thử xoay sản phẩm
                        if not individual and stock_w >= prod_h and stock_h >= prod_w:
                            for x_pos in range(stock_w - prod_h + 1):
                                for y_pos in range(stock_h - prod_w + 1):
                                    if self._can_place_(stock, (x_pos, y_pos), (prod_h, prod_w)):
                                        individual.append({"stock_idx": stock_idx, "size": (prod_h, prod_w), "position": (x_pos, y_pos)})
                                        break
                                if individual:
                                    break
            if individual:
                population.append(individual)

        # Nếu quần thể không có cá thể, thêm một cá thể mặc định
        if not population:
            population.append([{"stock_idx": -1, "size": (0, 0), "position": (0, 0)}])

        return population

    def _evaluate_solution(self, solution, products, stocks):
        """
        Đánh giá fitness của một giải pháp: Diện tích lãng phí.
        """
        total_filled = 0
        total_area = 0

        # Tính diện tích tổng cộng của tất cả các stock
        for stock_idx, stock in enumerate(stocks):
            stock_w, stock_h = self._get_stock_size_(stock)
            total_area += stock_w * stock_h

        # Tính diện tích đã được sử dụng cho các sản phẩm trong giải pháp
        for item in solution:
            prod_w, prod_h = item["size"]
            total_filled += prod_w * prod_h

        # Trả về tỷ lệ diện tích lãng phí, muốn giảm diện tích lãng phí
        return 1 - (total_filled / total_area)

    def _select_population(self, population, fitness_scores):
        """
        Lựa chọn các cá thể tốt nhất từ quần thể dựa trên fitness scores.
        """
        sorted_population = [x for _, x in sorted(zip(fitness_scores, population), key = lambda pair: pair[0])]
        return sorted_population[:self.population_size // 2]

    def _crossover_and_mutate(self, population, products, stocks):
        """
        Lai ghép và đột biến để tạo ra thế hệ tiếp theo.
        """
        next_generation = []
        for _ in range(len(population)):
            parent1, parent2 = random.sample(population, 2)
            child = []

            # Lai ghép: kết hợp các sản phẩm theo thứ tự ưu tiên
            for i in range(len(parent1)):
                if i < len(parent2) and np.random.rand() > 0.5:
                    child.append(parent2[i])
                else:
                    child.append(parent1[i])

            # Đột biến: thay đổi vị trí của một sản phẩm ngẫu nhiên
            if child and np.random.rand() < self.mutation_rate:
                mutation_idx = np.random.randint(0, len(child))
                mutated_item = child[mutation_idx]
                stock_idx = mutated_item["stock_idx"]
                stock = stocks[stock_idx]
                stock_w, stock_h = self._get_stock_size_(stock)
                prod_w, prod_h = mutated_item["size"]

                # Tìm vị trí mới từ trên xuống dưới, từ trái sang phải để đặt sản phẩm
                for x_pos in range(stock_w - prod_w + 1):
                    for y_pos in range(stock_h - prod_h + 1):
                        if self._can_place_(stock, (x_pos, y_pos), (prod_w, prod_h)):
                            mutated_item["position"] = (x_pos, y_pos)
                            break
                    if mutated_item["position"] != child[mutation_idx]["position"]:
                        break

                child[mutation_idx] = mutated_item

            next_generation.append(child)

        # Nếu không có thế hệ mới, giữ lại quần thể cũ
        if not next_generation:
            return population[:1]

        return next_generation