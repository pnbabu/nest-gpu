/*
 *  connect_rules.h
 *
 *  This file is part of NEST GPU.
 *
 *  Copyright (C) 2021 The NEST Initiative
 *
 *  NEST GPU is free software: you can redistribute it and/or modify
 *  it under the terms of the GNU General Public License as published by
 *  the Free Software Foundation, either version 2 of the License, or
 *  (at your option) any later version.
 *
 *  NEST GPU is distributed in the hope that it will be useful,
 *  but WITHOUT ANY WARRANTY; without even the implied warranty of
 *  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 *  GNU General Public License for more details.
 *
 *  You should have received a copy of the GNU General Public License
 *  along with NEST GPU.  If not, see <http://www.gnu.org/licenses/>.
 *
 */

#ifndef CONNECTRULES_H
#define CONNECTRULES_H
#ifdef _OPENMP
#include <omp.h>
#define THREAD_MAXNUM omp_get_max_threads()
#define THREAD_IDX omp_get_thread_num()
#else
#endif
#ifdef _OPENMP
  omp_lock_t *lock = new omp_lock_t[n_source];
  for (int i=0; i<n_source; i++) {
    omp_init_lock(&(lock[i]));
  }
#pragma omp parallel for default(shared) collapse(2)
#endif
#ifdef _OPENMP
      omp_set_lock(&(lock[isn]));
#endif
#ifdef _OPENMP
      omp_unset_lock(&(lock[isn]));
#endif
#ifdef _OPENMP
  delete[] lock;
#endif
#ifdef _OPENMP
  omp_lock_t *lock = new omp_lock_t[n_source];
  for (int i=0; i<n_source; i++) {
    omp_init_lock(&(lock[i]));
  }
#pragma omp parallel for default(shared)
#endif
#ifdef _OPENMP
    omp_set_lock(&(lock[isn]));
#endif
#ifdef _OPENMP
      omp_unset_lock(&(lock[isn]));
#endif
#ifdef _OPENMP
  delete[] lock;
#endif
#ifdef _OPENMP
  omp_lock_t *lock = new omp_lock_t[n_source];
  for (int i=0; i<n_source; i++) {
    omp_init_lock(&(lock[i]));
  }
#endif
#ifdef _OPENMP
#pragma omp parallel for default(shared)
#endif
#ifdef _OPENMP
	  omp_set_lock(&(lock[isn]));
#endif
#ifdef _OPENMP
	  omp_unset_lock(&(lock[isn]));
#endif
#ifdef _OPENMP
  delete[] lock;
#endif
#ifdef _OPENMP
  omp_lock_t *lock = new omp_lock_t[n_source];
  for (int i=0; i<n_source; i++) {
    omp_init_lock(&(lock[i]));
  }
#endif
#ifdef _OPENMP
#pragma omp parallel for default(shared)
#endif
#ifdef _OPENMP
	  omp_set_lock(&(lock[isn]));
#endif	  
#ifdef _OPENMP
	  omp_unset_lock(&(lock[isn]));
#endif
#ifdef _OPENMP
  delete[] lock;
#endif

#endif
