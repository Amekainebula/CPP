//#define _CRT_SECURE_NO_WARNINGS 1
//#include <bits/stdc++.h>
//#define int long long
//#define double long double
//#define ull unsigned long long
//#define endl '\n'
//using namespace std;
//int t, n, m, k;
//int a[10000005];
//set<int> s;
//signed main()
//{
//	cin >> t;
//	while (t--)
//	{
//		s.clear();
//		cin >> n >> m >> k;
//		for (int i = 1; i <= m; i++)
//			cin >> a[i];
//		for (int i = 1; i <= k; i++)
//		{
//			int x;
//			cin >> x;
//			s.insert(x);
//		}
//		if (n - k == 0)
//		{
//			for (int i = 1; i <= m; i++)
//				cout << "1";
//			cout << endl;
//			continue;
//		}
//		if (n - k > 1)
//		{
//			for (int i = 1; i <= m; i++)
//				cout << "0";
//			cout << endl;
//			continue;
//		}
//		int missing = 0;
//		for (int i = 1; i <= n; i++)
//		{
//			if (s.find(i) == s.end())
//			{
//				missing = i;
//				break;
//			}
//		}
//		if (missing == 0)
//		{
//			for (int i = 1; i <= m; i++)
//				cout << "1";
//			cout << endl;
//			continue;
//		}
//		for (int i = 1; i <= m; i++)
//		{
//			if (a[i] == missing) 
//			{
//				cout << "1";
//			}
//			else
//			{
//				cout << "0";
//			}
//		}
//		cout << endl;
//	}
//	return 0;
//}