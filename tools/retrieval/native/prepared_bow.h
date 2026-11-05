// SPDX-License-Identifier: BSD-3-Clause
// Stage adapter for COLMAP's unchanged BoW+HE scoring rules.
// COLMAP-derived portions retain the notices in LICENSES/COLMAP.txt; see NOTICE.
#pragma once

#include <algorithm>
#include <atomic>
#include <chrono>
#include <fstream>
#include <iomanip>
#include <map>
#include <numeric>
#include <sstream>
#include <omp.h>

namespace colmap {
namespace benchmark {

using Encoded = retrieval::VisualIndex::EncodedImage;
using Clock = std::chrono::steady_clock;

inline double Elapsed(Clock::time_point start) {
  return std::chrono::duration<double>(Clock::now() - start).count();
}

template <typename T>
void Transfer(std::ostream& stream, const T& value) {
  stream.write(reinterpret_cast<const char*>(&value), sizeof(T));
  THROW_CHECK(stream.good());
}

template <typename T>
void Transfer(std::istream& stream, T* value) {
  stream.read(reinterpret_cast<char*>(value), sizeof(T));
  THROW_CHECK(stream.good());
}

struct Record {
  int id = -1;
  std::string name;
  Encoded encoded;
};

inline void Write(std::ostream& stream, const Record& record) {
  Transfer(stream, record.id);
  Transfer(stream, static_cast<uint64_t>(record.name.size()));
  stream.write(record.name.data(), record.name.size());
  Transfer(stream, record.encoded.rows);
  Transfer(stream, record.encoded.neighbors);
  for (auto word : record.encoded.words) Transfer(stream, word);
  for (auto signature : record.encoded.signatures) Transfer(stream, signature);
  Transfer(stream, static_cast<uint64_t>(record.encoded.geometry.size()));
  for (const auto& geometry : record.encoded.geometry) {
    Transfer(stream, geometry.x);
    Transfer(stream, geometry.y);
    Transfer(stream, geometry.scale);
    Transfer(stream, geometry.orientation);
  }
}

inline Record Read(std::istream& stream) {
  Record record;
  uint64_t length;
  Transfer(stream, &record.id);
  Transfer(stream, &length);
  THROW_CHECK_LE(length, 1 << 20);
  record.name.resize(length);
  stream.read(record.name.data(), length);
  Transfer(stream, &record.encoded.rows);
  Transfer(stream, &record.encoded.neighbors);
  THROW_CHECK_GE(record.encoded.rows, 0);
  THROW_CHECK_LE(record.encoded.rows, 1000000);
  THROW_CHECK_GT(record.encoded.neighbors, 0);
  THROW_CHECK_LE(record.encoded.neighbors, 32);
  const size_t size = static_cast<size_t>(record.encoded.rows) * record.encoded.neighbors;
  record.encoded.words.resize(size);
  record.encoded.signatures.resize(size);
  for (auto& word : record.encoded.words) Transfer(stream, &word);
  for (auto& signature : record.encoded.signatures) Transfer(stream, &signature);
  Transfer(stream, &length);
  THROW_CHECK(length == 0 || length == static_cast<uint64_t>(record.encoded.rows));
  record.encoded.geometry.resize(length);
  for (auto& geometry : record.encoded.geometry) {
    Transfer(stream, &geometry.x);
    Transfer(stream, &geometry.y);
    Transfer(stream, &geometry.scale);
    Transfer(stream, &geometry.orientation);
  }
  return record;
}

inline void Header(std::ostream& stream, uint64_t count) {
  Transfer(stream, uint64_t{0x474d4c424f573031});
  Transfer(stream, count);
}

inline uint64_t Header(std::istream& stream) {
  uint64_t magic, count;
  Transfer(stream, &magic);
  Transfer(stream, &count);
  THROW_CHECK_EQ(magic, uint64_t{0x474d4c424f573031});
  THROW_CHECK_LE(count, 10000000);
  return count;
}

inline void Metrics(const std::filesystem::path& path,
                    const std::map<std::string, double>& values,
                    const std::vector<double>& trials = {},
                    bool complete = true,
                    const std::string& query_identity = "") {
  const auto temporary = std::filesystem::path(path.string() + ".writing");
  std::ofstream stream(temporary);
  THROW_CHECK_FILE_OPEN(stream, path);
  stream << std::setprecision(16) << "{\n  \"protocol\": \"prepared_bow_v1\"";
  for (const auto& item : values) stream << ",\n  \"" << item.first << "\": " << item.second;
  stream << ",\n  \"query_trials_seconds\": [";
  for (size_t i = 0; i < trials.size(); ++i) {
    if (i) stream << ',';
    stream << trials[i];
  }
  stream << "],\n  \"complete\": " << (complete ? "true" : "false")
         << ",\n  \"query_identity\": \"" << query_identity << "\"\n}\n";
  stream.close();
  THROW_CHECK(!stream.fail());
  std::filesystem::rename(temporary, path);
}

inline int Run(const std::string& stage,
               const std::filesystem::path& database_path,
               const std::filesystem::path& vocabulary,
               const std::filesystem::path& image_list,
               const std::filesystem::path& directory,
               const std::filesystem::path& output_index,
               const std::filesystem::path& metrics_path,
               int threads, int topk, int repeats, int warmup_queries,
               const std::string& previous_trials, const std::string& query_identity) {
  THROW_CHECK_GT(threads, 0);
  omp_set_num_threads(threads);
  THROW_CHECK_GT(repeats, 0);
  std::filesystem::create_directories(directory);
  const auto start = Clock::now();
  auto index = retrieval::VisualIndex::Read(vocabulary);
  double startup = Elapsed(start);
  retrieval::VisualIndex::QueryOptions options;
  options.num_threads = 1;
  options.num_neighbors = 5;
  options.num_checks = 64;
  options.num_images_after_verification = 0;
  options.max_num_images = topk;
  const auto begin = Clock::now();

  if (stage == "encode") {
    auto database = Database::Open(database_path);
    auto images = database->ReadAllImages();
    if (!image_list.empty()) {
      const auto names = ReadTextFileLines(image_list);
      const FlatHashSet<std::string> selected(names.begin(), names.end());
      images.erase(std::remove_if(images.begin(), images.end(), [&](const Image& image) {
        return !selected.count(image.Name());
      }), images.end());
      THROW_CHECK_EQ(images.size(), names.size());
    }
    std::sort(images.begin(), images.end(), [](const Image& a, const Image& b) {
      return a.Name() < b.Name();
    });
    std::ofstream db_file(directory / "database.partial.bin", std::ios::binary);
    std::ofstream query_file(directory / "queries.partial.bin", std::ios::binary);
    Header(db_file, images.size());
    Header(query_file, images.size());
    double read_seconds = 0, db_seconds = 0, query_seconds = 0, write_seconds = 0;
    for (size_t offset = 0; offset < images.size(); offset += threads) {
      const size_t count = std::min<size_t>(threads, images.size() - offset);
      auto timer = Clock::now();
      std::vector<FeatureKeypoints> points(count);
      std::vector<FeatureDescriptorsFloat> descriptors(count);
      std::vector<Record> db_records(count), query_records(count);
      for (size_t i = 0; i < count; ++i) {
        points[i] = database->ReadKeypoints(images[offset + i].ImageId());
        descriptors[i] = database->ReadDescriptors(images[offset + i].ImageId()).ToFloat();
        db_records[i].id = query_records[i].id = images[offset + i].ImageId();
        db_records[i].name = query_records[i].name = images[offset + i].Name();
      }
      read_seconds += Elapsed(timer);
      timer = Clock::now();
#pragma omp parallel for num_threads(threads) schedule(static)
      for (int i = 0; i < static_cast<int>(count); ++i) {
        auto db_options = options;
        db_options.num_neighbors = 1;
        db_records[i].encoded = index->Encode(db_options, points[i], descriptors[i]);
      }
      db_seconds += Elapsed(timer);
      timer = Clock::now();
#pragma omp parallel for num_threads(threads) schedule(static)
      for (int i = 0; i < static_cast<int>(count); ++i) {
        query_records[i].encoded = index->Encode(options, {}, descriptors[i]);
      }
      query_seconds += Elapsed(timer);
      timer = Clock::now();
      for (size_t i = 0; i < count; ++i) {
        Write(db_file, db_records[i]);
        Write(query_file, query_records[i]);
      }
      write_seconds += Elapsed(timer);
      LOG(INFO) << "Encoded " << offset + count << "/" << images.size();
    }
    db_file.close();
    query_file.close();
    std::filesystem::rename(directory / "database.partial.bin", directory / "database.bin");
    std::filesystem::rename(directory / "queries.partial.bin", directory / "queries.bin");
    Metrics(metrics_path, {{"images", images.size()}, {"startup_seconds", startup},
        {"input_read_seconds", read_seconds}, {"database_encode_seconds", db_seconds},
        {"query_encode_seconds", query_seconds}, {"cache_write_seconds", write_seconds},
        {"compute_seconds", db_seconds + query_seconds}, {"stage_wall_seconds", Elapsed(begin)}});
    return EXIT_SUCCESS;
  }

  if (stage == "build") {
    std::ifstream stream(directory / "database.bin", std::ios::binary);
    const auto count = Header(stream);
    double read_seconds = 0, core_seconds = 0;
    for (uint64_t i = 0; i < count; ++i) {
      auto timer = Clock::now();
      auto record = Read(stream);
      read_seconds += Elapsed(timer);
      timer = Clock::now();
      index->AddEncoded(record.id, record.encoded);
      core_seconds += Elapsed(timer);
    }
    auto timer = Clock::now();
    index->Prepare();
    core_seconds += Elapsed(timer);
    timer = Clock::now();
    index->Write(output_index);
    const double write_seconds = Elapsed(timer);
    Metrics(metrics_path, {{"images", count}, {"startup_seconds", startup},
        {"input_read_seconds", read_seconds}, {"compute_seconds", core_seconds},
        {"cache_write_seconds", write_seconds}, {"stage_wall_seconds", Elapsed(begin)}});
    return EXIT_SUCCESS;
  }

  THROW_CHECK(stage == "query" || stage == "verify");
  index->Prepare();
  std::ifstream stream(directory / "queries.bin", std::ios::binary);
  const auto count = Header(stream);
  std::vector<Record> queries;
  queries.reserve(count);
  for (uint64_t i = 0; i < count; ++i) queries.push_back(Read(stream));
  const double read_seconds = Elapsed(begin);
  auto timer = Clock::now();
#pragma omp parallel for num_threads(threads) schedule(static)
  for (int i = 0; i < static_cast<int>(count); ++i) index->FinalizeQuery(&queries[i].encoded);
  const double finalize_seconds = Elapsed(timer);
  std::vector<std::vector<retrieval::ImageScore>> scores(count);

  if (stage == "verify") {
    auto database = Database::Open(database_path);
    double max_error = 0;
    std::ofstream verification(directory / "verification_scores.tsv");
    verification << std::setprecision(16);
    for (size_t i = 0; i < count; ++i) {
      std::vector<retrieval::ImageScore> reference;
      index->Query(options, database->ReadKeypoints(queries[i].id),
          database->ReadDescriptors(queries[i].id).ToFloat(), &reference);
      index->QueryEncoded(options, queries[i].encoded, &scores[i]);
      THROW_CHECK_EQ(reference.size(), scores[i].size());
      for (const auto& score : scores[i]) {
        verification << queries[i].name << '\t' << score.image_id << '\t' << score.score << '\n';
      }
      for (size_t j = 0; j < reference.size(); ++j) {
        THROW_CHECK_EQ(reference[j].image_id, scores[i][j].image_id);
        max_error = std::max<double>(max_error, std::abs(reference[j].score - scores[i][j].score));
      }
    }
    THROW_CHECK_LE(max_error, 1e-6);
    Metrics(metrics_path, {{"images", count}, {"max_score_error", max_error}});
    return EXIT_SUCCESS;
  }

  std::vector<double> trials;
  std::istringstream previous(previous_trials);
  std::string value;
  while (std::getline(previous, value, ',')) {
    const double seconds = std::stod(value);
    THROW_CHECK_GT(seconds, 0.0);
    trials.push_back(seconds);
  }
  THROW_CHECK_LE(trials.size(), static_cast<size_t>(repeats));
  THROW_CHECK_GE(warmup_queries, 0);
  const size_t warmup_count = warmup_queries == 0 ? count :
      std::min<uint64_t>(count, warmup_queries);
  const int first_measured_trial = trials.size() + 1;
  double warmup_seconds = 0;
  auto checkpoint = [&](int trial) {
    Metrics(metrics_path, {{"images", count}, {"startup_seconds", startup},
        {"input_read_seconds", read_seconds}, {"query_finalize_seconds", finalize_seconds},
        {"threads", threads}, {"warmup_queries", warmup_count},
        {"warmup_seconds", warmup_seconds}, {"last_completed_trial", trial},
        {"requested_repeats", repeats}}, trials, trials.size() == static_cast<size_t>(repeats),
        query_identity);
  };
  auto export_pairs = [&]() {
    std::map<int, std::string> names;
    for (const auto& query : queries) names[query.id] = query.name;
    const auto destination = directory / "ranked_pairs.txt";
    const auto temporary = directory / "ranked_pairs.partial.txt";
    std::ofstream pairs(temporary);
    THROW_CHECK_FILE_OPEN(pairs, temporary);
    for (size_t i = 0; i < count; ++i) {
      for (const auto& score : scores[i]) {
        if (score.image_id != queries[i].id) {
          pairs << queries[i].name << ' ' << names.at(score.image_id) << '\n';
        }
      }
    }
    pairs.close();
    THROW_CHECK(!pairs.fail());
    std::filesystem::rename(temporary, destination);
  };
  // Persist completed rounds outside the search timer. An interrupted next round
  // leaves the previous full pairs and timings intact.
  for (int trial = 0; trial <= repeats; ++trial) {
    if (trial > 0 && trial < first_measured_trial) continue;
    const size_t query_count = trial == 0 ? warmup_count : count;
    LOG(INFO) << "Starting retrieval trial " << trial << ": " << query_count << " queries";
    timer = Clock::now();
    std::atomic<size_t> completed{0};
#pragma omp parallel for num_threads(threads) schedule(dynamic, 1)
    for (int i = 0; i < static_cast<int>(query_count); ++i) {
      index->QueryEncoded(options, queries[i].encoded, &scores[i]);
      const auto done = ++completed;
      const size_t interval = std::min<uint64_t>(64, std::max<uint64_t>(1, query_count / 100));
      if (done % interval == 0 || done == query_count) {
        LOG(INFO) << "Retrieval trial " << trial << ": " << done << "/" << query_count;
      }
    }
    const double seconds = Elapsed(timer);
    LOG(INFO) << "Pure retrieval trial " << trial << ": " << seconds << "s";
    if (trial) {
      trials.push_back(seconds);
      export_pairs();
    } else {
      warmup_seconds = seconds;
    }
    checkpoint(trial);
    LOG(INFO) << "Saved retrieval checkpoint after trial " << trial;
  }
  return EXIT_SUCCESS;
}

}  // namespace benchmark
}  // namespace colmap
