//#define _CRT_SECURE_NO_WARNINGS 1
//#include <bits/stdc++.h>
//#define ll long long
//#define ld long double
//#define ull unsigned long long
//#define endl '\n'
//using namespace std;
//ll n, k, ans = 0;
//map<ll, ll> mp;
//ll a[1000005];
//set<ll> s;
//ll ok = 1;
//void solve()
//{
//    mp.clear();
//    s.clear();
//    ok = 1;
//    cin >> n >> k;
//    for (int i = 1; i <= n; i++)
//    {
//        ll x;
//        cin >> x;
//        s.insert(x);
//        mp[x]++;
//        if (mp[x] > ok)ok = 0;
//   }
//    if (!k)
//    {
//        cout << s.size() << endl;
//        return;
//    }
//    if (ok)
//    {
//        ans = s.size() - k;
//        if (ans <= 0)ans = 1;
//        cout << ans << endl;
//        return;
//    }
//    ll cnt = 0;
//    ll temp = s.size();
//    for (auto it : s)
//    {
//        a[cnt++] = mp[it];
//    }
//    sort(a, a + cnt);
//    for (int i = 0; i < cnt; i++)
//    {
//        if (k >= a[i])
//        {
//            temp--;
//            k -= a[i];
//        }
//        else
//            break;
//    }
//    ans = temp;
//    if (ans <= 0)ans = 1;
//    cout << ans << endl;
//    /*for (int i = 0; i < 10000005; i++)
//    {
//        if (a[i] == 0)continue;
//        if (k >= a[i])
//        {
//            a[i] = 0;
//            temp--;
//            k -= a[i];
//        }
//        else
//            break;
//    }
//    ans = temp;
//    if (ans <= 0)ans = 1;
//    cout << ans << endl;
//    */
//}
//int main()
//{
//    ios::sync_with_stdio(false);
//    cin.tie(0);
//    ll T;
//    cin >> T;
//    //T = 1;
//    while (T--)
//    {
//        solve();
//    }
//    return 0;
//}