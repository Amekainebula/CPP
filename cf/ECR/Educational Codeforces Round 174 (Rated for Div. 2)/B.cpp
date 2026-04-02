//#include <bits/stdc++.h>
//#define int long long
//#define ld long double
//#define ull unsigned long long
//#define lowbit(x) (x & -x)
//#define pb push_back
//#define pii pair<int, int>
//#define mpr make_pair
//#define fi first
//#define se second
//#define all(x) x.begin(), x.end()
//#define rall(x) x.rbegin(), x.rend()
//#define sz(x) (int)(x).size()
//#define INF 0x7fffffffffffffff
//#define endl '\n' 
//#define yes cout << "YES"<< endl
//#define no cout << "NO"<< endl
//#define yess cout << "Yes" << endl
//#define noo cout << "No"<< endl
//using namespace std;
//int maps[705][705];
////int vis[705][705];
//int net[2][2] = { {1,0},{0,1} };
//set<int>s1, s2;
//int n, m;
//void solve()
//{
//    
//    cin >> n >> m;
//    //bool flag = 0;
//    //memset(vis, 0, sizeof(vis));
//    s1.clear();
//    s2.clear();
//    for (int i = 1; i <= n; i++)
//    {
//        for (int j = 1; j <= m; j++)
//        {
//            cin >> maps[i][j];
//            s1.insert(maps[i][j]);
//        }
//    }
//    for (int i = 1; i <= n; i++)
//    {
//        for (int j = 1; j <= m; j++)
//        {
//            for (int k = 0; k < 2; k++)
//            {
//                int x = i + net[k][0];
//                int y = j + net[k][1];
//                if (x > 0 && x <= n && y > 0 && y <= m )
//                {
//                    if (maps[x][y] == maps[i][j])
//                        s2.insert(maps[x][y]);
//                }
//            }
//        }
//    }
//    if (s2.size() != 0)
//    {
//        cout << sz(s2) + sz(s1) - 2 << endl;
//    }
//    else
//    {
//        cout << sz(s1) + sz(s2) - 1 << endl;
//    }
//}
//signed main()
//{
//    ios::sync_with_stdio(false);
//    cin.tie(0);
//    cout.tie(0);
//    int T = 1;
//    cin >> T;
//    while (T--)
//    {
//        solve();
//    }
//    return 0;
//}